from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import os
import json
from io import StringIO
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="FairSight AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Configure Gemini ───────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_model = None

def get_gemini_model():
    global gemini_model
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE" and not gemini_model:
        try:
            import google.generativeai as genai
        except ImportError:
            return None

        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    return gemini_model


# ─── Utility Functions ───────────────────────────────────────────────────────

def encode_dataframe(df: pd.DataFrame):
    """Encode all categorical columns, return encoded df + encoders."""
    df = df.copy()
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    le_dict = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict


def get_column_info(df: pd.DataFrame):
    """Return column names, types, and sample unique values."""
    info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        uniq = df[col].dropna().unique()[:8].tolist()
        # Make JSON serializable
        uniq = [str(v) if not isinstance(v, (int, float, bool)) else v for v in uniq]
        info.append({
            "name": col,
            "dtype": dtype,
            "unique_count": int(df[col].nunique()),
            "sample_values": uniq,
            "null_count": int(df[col].isnull().sum())
        })
    return info


def calculate_fairness_metrics(
    df,
    target_col,
    sensitive_col,
    classifier_name="random_forest",
    drop_sensitive_from_features=False,
    threshold=0.5,
):
    """Train model and calculate comprehensive fairness metrics."""
    df_enc, le_dict = encode_dataframe(df)
    
    X = df_enc.drop(columns=[target_col])
    if drop_sensitive_from_features and sensitive_col in X.columns:
        X = X.drop(columns=[sensitive_col])
    y = df_enc[target_col]
    sensitive = df_enc[sensitive_col]
    
    # Choose classifier
    classifiers = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }
    clf = classifiers.get(classifier_name, classifiers["random_forest"])
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.25, random_state=42, stratify=y
    )

    # Scale features to stabilize training and improve baseline accuracy.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        y_pred = (clf.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
    
    # Core fairness metrics
    dp_diff = float(demographic_parity_difference(y_test, y_pred, sensitive_features=s_test))
    eq_odds_diff = float(equalized_odds_difference(y_test, y_pred, sensitive_features=s_test))
    
    # MetricFrame for per-group breakdown
    mf = MetricFrame(
        metrics={
            "accuracy": lambda yt, yp: float(accuracy_score(yt, yp)),
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=s_test
    )
    
    # Convert group metrics to JSON-safe dicts
    group_data = {}
    by_group = mf.by_group
    
    # Map encoded values back to original labels if possible
    sensitive_original = df[sensitive_col]
    group_labels = {}
    if sensitive_col in le_dict:
        le = le_dict[sensitive_col]
        for enc_val in s_test.unique():
            try:
                orig = le.inverse_transform([enc_val])[0]
                group_labels[enc_val] = str(orig)
            except:
                group_labels[enc_val] = str(enc_val)
    else:
        for v in s_test.unique():
            group_labels[v] = str(v)
    
    for metric_name in ["accuracy", "selection_rate", "false_positive_rate", "false_negative_rate"]:
        group_data[metric_name] = {
            group_labels.get(k, str(k)): round(float(v), 4)
            for k, v in by_group[metric_name].items()
        }
    
    disparate_impact = round(1 - abs(dp_diff), 4)
    overall_acc = round(float(accuracy_score(y_test, y_pred)), 4)
    
    # Feature importance (if available)
    feature_importance = {}
    if hasattr(clf, 'feature_importances_'):
        fi = clf.feature_importances_
        cols = X.columns.tolist()
        feature_importance = {cols[i]: round(float(fi[i]), 4) for i in np.argsort(fi)[-8:][::-1]}
    elif hasattr(clf, 'coef_'):
        coefs = np.abs(clf.coef_[0])
        cols = X.columns.tolist()
        feature_importance = {cols[i]: round(float(coefs[i]), 4) for i in np.argsort(coefs)[-8:][::-1]}
    
    return {
        "disparate_impact_ratio": disparate_impact,
        "statistical_parity_difference": round(dp_diff, 4),
        "equal_opportunity_difference": round(eq_odds_diff, 4),
        "overall_accuracy": overall_acc,
        "group_metrics": group_data,
        "feature_importance": feature_importance,
        "model_used": classifier_name,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }


def get_severity(metrics: dict) -> str:
    di = metrics["disparate_impact_ratio"]
    spd = abs(metrics["statistical_parity_difference"])
    if di < 0.7 or spd > 0.2:
        return "Critical"
    elif di < 0.8 or spd > 0.1:
        return "High"
    elif di < 0.9 or spd > 0.05:
        return "Medium"
    return "Low"


def get_mitigation_suggestion(metrics: dict, severity: str, sensitive_col: str, classifier_name: str):
    """Generate actionable bias mitigation suggestions."""
    suggestions = []

    if metrics["disparate_impact_ratio"] < 0.8:
        suggestions.append({
            "title": "Apply Reweighing (AIF360)",
            "description": "Balance training examples so underrepresented groups matter more during fitting.",
            "code": f"from aif360.algorithms.preprocessing import Reweighing\nreweigh = Reweighing(unprivileged_groups=[{{'{sensitive_col}': 0}}], privileged_groups=[{{'{sensitive_col}': 1}}])",
            "difficulty": "Medium",
        })

    if abs(metrics["statistical_parity_difference"]) > 0.1:
        suggestions.append({
            "title": "Tune Decision Threshold",
            "description": "Adjust the prediction cutoff to reduce the parity gap across groups.",
            "code": "y_pred = (model.predict_proba(X_test)[:, 1] >= 0.35).astype(int)",
            "difficulty": "Easy",
        })

    if metrics["overall_accuracy"] < 0.7 or classifier_name != "random_forest":
        suggestions.append({
            "title": "Use Random Forest by Default",
            "description": "Replace LogisticRegression with a stronger tree ensemble for a better baseline model.",
            "code": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)",
            "difficulty": "Easy",
        })

    if severity in ("High", "Critical"):
        suggestions.append({
            "title": "Collect More Balanced Data",
            "description": "Add more samples from underrepresented groups before retraining.",
            "code": "# Add more rows for the underrepresented group before model.fit()",
            "difficulty": "Medium",
        })
        suggestions.append({
            "title": f"Drop '{sensitive_col}' Proxies",
            "description": "Remove features that indirectly leak the protected attribute.",
            "code": f"X = X.drop(columns=['{sensitive_col}'])",
            "difficulty": "Easy",
        })

    if not suggestions:
        suggestions.append({
            "title": "Monitor and Recheck",
            "description": "The model looks acceptable now, but keep tracking fairness on new data.",
            "code": "# Re-run bias audit whenever training data changes",
            "difficulty": "Easy",
        })

    return suggestions


def build_mitigation_comparison(df: pd.DataFrame, target_col: str, sensitive_col: str, classifier_name: str):
    """Compare baseline fairness metrics to a simple mitigation pass."""
    before = calculate_fairness_metrics(df, target_col, sensitive_col, classifier_name)
    after = calculate_fairness_metrics(
        df,
        target_col,
        sensitive_col,
        classifier_name,
        drop_sensitive_from_features=True,
        threshold=0.35,
    )

    return {
        "before": before,
        "after": after,
        "before_severity": get_severity(before),
        "after_severity": get_severity(after),
        "before_fairness_score": round(
            (min(before["disparate_impact_ratio"], 1) * 40)
            + (max(0, 1 - abs(before["statistical_parity_difference"]) * 5) * 30)
            + (max(0, 1 - abs(before["equal_opportunity_difference"]) * 5) * 30)
        ),
        "after_fairness_score": round(
            (min(after["disparate_impact_ratio"], 1) * 40)
            + (max(0, 1 - abs(after["statistical_parity_difference"]) * 5) * 30)
            + (max(0, 1 - abs(after["equal_opportunity_difference"]) * 5) * 30)
        ),
        "strategy": {
            "title": "Remove sensitive feature + lower threshold",
            "description": f"Retrain without '{sensitive_col}' in the feature set and use a stricter threshold to reduce disparity.",
        },
        "mitigation_notes": [
            f"Removed '{sensitive_col}' from the training feature set.",
            "Applied threshold tuning (0.50 -> 0.35) for the mitigation pass.",
            f"Fairness Score changed from {round((min(before['disparate_impact_ratio'], 1) * 40) + (max(0, 1 - abs(before['statistical_parity_difference']) * 5) * 30) + (max(0, 1 - abs(before['equal_opportunity_difference']) * 5) * 30))} to {round((min(after['disparate_impact_ratio'], 1) * 40) + (max(0, 1 - abs(after['statistical_parity_difference']) * 5) * 30) + (max(0, 1 - abs(after['equal_opportunity_difference']) * 5) * 30))}.",
            f"Accuracy changed from {before['overall_accuracy']} to {after['overall_accuracy']}.",
        ],
    }


def get_gemini_explanation(metrics: dict, sensitive_col: str, target_col: str, dataset_name: str = "uploaded") -> dict:
    """Call Gemini for structured explanation. Returns dict with sections."""
    model = get_gemini_model()
    
    if not model:
        # Fallback explanation when no API key
        severity = get_severity(metrics)
        di = metrics["disparate_impact_ratio"]
        spd = metrics["statistical_parity_difference"]
        
        if severity in ("Critical", "High"):
            verdict = "HIGH BIAS - needs mitigation"
            action = f"Apply reweighing or remove '{sensitive_col}'-linked proxy features before retraining."
        elif severity == "Medium":
            verdict = "MODERATE BIAS - monitor closely"
            action = f"Tune the decision threshold and rebalance samples for '{sensitive_col}'."
        else:
            verdict = "LOW BIAS - within acceptable thresholds"
            action = "Continue monitoring as new data arrives."
        
        return {
            "summary": f"{verdict}. Disparate Impact Ratio is {di} and Statistical Parity Difference is {abs(spd):.3f} for '{sensitive_col}'.",
            "disadvantaged_groups": f"Groups with lower selection rates or accuracy on '{sensitive_col}' are most affected. Check the group charts for specifics.",
            "root_cause": f"Historical bias and proxy variables in the dataset likely influence '{target_col}' through '{sensitive_col}'.",
            "recommendation": action,
            "gemini_powered": False
        }
    
    prompt = f"""
You are an AI fairness expert. Analyze these bias audit results and respond in JSON format only (no markdown, no backticks).

Dataset: {dataset_name}
Sensitive attribute (protected group): {sensitive_col}
Target variable (what model predicts): {target_col}

Fairness Metrics:
- Disparate Impact Ratio: {metrics['disparate_impact_ratio']} (fair if >= 0.8)
- Statistical Parity Difference: {metrics['statistical_parity_difference']} (fair if <= 0.1)
- Equal Opportunity Difference: {metrics['equal_opportunity_difference']}
- Overall Model Accuracy: {metrics['overall_accuracy']}
- Per-group metrics: {json.dumps(metrics['group_metrics'], indent=2)}

CRITICAL RULES:
1. If Disparate Impact < 0.8 OR Statistical Parity > 0.1, say bias exists.
2. Do not say the model is fair if bias exists.
3. Recommendation must mention a concrete fix like reweighing, threshold tuning, or removing proxy features.

Return this exact JSON structure:
 {{
  "summary": "2-3 sentence plain-English verdict on bias level and key finding",
  "disadvantaged_groups": "Which specific groups are most disadvantaged and how",
  "root_cause": "Most likely cause of this bias in plain terms (data collection, historical discrimination, proxy variables, etc.)",
  "recommendation": "One concrete, actionable technical fix with specifics",
  "gemini_powered": true
 }}
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Remove any markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        result["gemini_powered"] = True
        return result
    except Exception as e:
        return {
            "summary": f"HIGH BIAS - needs mitigation. Disparate Impact: {metrics['disparate_impact_ratio']}, Statistical Parity Diff: {metrics['statistical_parity_difference']:.4f}.",
            "disadvantaged_groups": "See group metrics chart for per-group breakdown.",
            "root_cause": "Could not generate detailed analysis. Check your Gemini API key.",
            "recommendation": "Apply reweighing, threshold tuning, or remove proxy features before retraining.",
            "gemini_powered": False,
            "error": str(e)
        }


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "FairSight AI API v2.0 — Running!",
        "gemini_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE"),
        "docs": "/docs"
    }


@app.post("/preview")
async def preview_dataset(file: UploadFile = File(...)):
    """Return column info and first 5 rows for column selection UI."""
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Dataset too small (need at least 10 rows)")
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Dataset needs at least 2 columns")
        
        return {
            "status": "ok",
            "rows": len(df),
            "columns": get_column_info(df),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")


@app.post("/audit")
async def audit_dataset(
    file: UploadFile = File(...),
    target_col: str = Query(..., description="Target column to predict"),
    sensitive_col: str = Query(..., description="Sensitive/protected attribute column"),
    classifier: str = Query("random_forest", description="Classifier: logistic, random_forest, decision_tree"),
):
    """Full bias audit: train model, compute fairness metrics, get Gemini explanation."""
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        df.columns = [str(c).strip() for c in df.columns]
        
        # Validate columns
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found. Available: {list(df.columns)}")
        if sensitive_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{sensitive_col}' not found. Available: {list(df.columns)}")
        if target_col == sensitive_col:
            raise HTTPException(status_code=400, detail="Target and sensitive columns must be different")
        
        # Drop rows with nulls in key columns
        df = df.dropna(subset=[target_col, sensitive_col])
        
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough clean rows for analysis (need 50+)")
        
        # Run analysis
        metrics = calculate_fairness_metrics(df, target_col, sensitive_col, classifier)
        severity = get_severity(metrics)
        explanation = get_gemini_explanation(metrics, sensitive_col, target_col, file.filename or "dataset")
        
        return {
            "status": "success",
            "severity": severity,
            "metrics": metrics,
            "explanation": explanation,
            "suggestions": get_mitigation_suggestion(metrics, severity, sensitive_col, classifier),
            "dataset_info": {
                "filename": file.filename,
                "total_rows": len(df),
                "target_col": target_col,
                "sensitive_col": sensitive_col,
                "target_distribution": df[target_col].value_counts().to_dict(),
                "sensitive_distribution": df[sensitive_col].value_counts().head(10).to_dict(),
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/mitigate")
async def mitigate_dataset(
    file: UploadFile = File(...),
    target_col: str = Query(..., description="Target column to predict"),
    sensitive_col: str = Query(..., description="Sensitive/protected attribute column"),
    classifier: str = Query("random_forest", description="Classifier: logistic, random_forest, decision_tree"),
):
    """Run a simple before/after mitigation comparison."""
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        df.columns = [str(c).strip() for c in df.columns]

        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found. Available: {list(df.columns)}")
        if sensitive_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{sensitive_col}' not found. Available: {list(df.columns)}")

        df = df.dropna(subset=[target_col, sensitive_col])
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough clean rows for mitigation comparison (need 50+)")

        comparison = build_mitigation_comparison(df, target_col, sensitive_col, classifier)

        return {
            "status": "success",
            **comparison,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "healthy", "gemini": bool(GEMINI_API_KEY)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FAIRSIGHT_PORT", "8001")))
