from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


def calculate_fairness_metrics(df, target_col, sensitive_col, classifier_name="logistic"):
    """Train model and calculate comprehensive fairness metrics."""
    df_enc, le_dict = encode_dataframe(df)
    
    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]
    sensitive = df_enc[sensitive_col]
    
    # Choose classifier
    classifiers = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }
    clf = classifiers.get(classifier_name, classifiers["logistic"])
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.25, random_state=42
    )
    
    clf.fit(X_train, y_train)
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


def get_mitigation_suggestion(metrics: dict, severity: str):
    """Generate actionable bias mitigation suggestions."""
    suggestions = []

    if metrics["disparate_impact_ratio"] < 0.8:
        suggestions.append("⚖️ Reweighing: balance training samples across groups before retraining.")

    if abs(metrics["statistical_parity_difference"]) > 0.1:
        suggestions.append("🎯 Threshold tuning: adjust the decision threshold for disadvantaged groups.")

    if severity in ("High", "Critical"):
        suggestions.append("🔄 Collect more data from underrepresented groups.")
        suggestions.append("🧹 Remove proxy features like zip code, school, or language that correlate with protected traits.")

    if not suggestions:
        suggestions.append("✅ Model looks fair right now. Keep monitoring as new data arrives.")

    return suggestions


def get_gemini_explanation(metrics: dict, sensitive_col: str, target_col: str, dataset_name: str = "uploaded") -> dict:
    """Call Gemini for structured explanation. Returns dict with sections."""
    model = get_gemini_model()
    
    if not model:
        # Fallback explanation when no API key
        severity = get_severity(metrics)
        di = metrics["disparate_impact_ratio"]
        spd = metrics["statistical_parity_difference"]
        
        if severity in ("Critical", "High"):
            bias_level = "significant bias"
            action = "Consider reweighing the training data or applying fairness constraints during model training."
        elif severity == "Medium":
            bias_level = "moderate bias"
            action = "Review data collection process and consider fairness-aware training techniques."
        else:
            bias_level = "low bias"
            action = "Continue monitoring bias metrics as new data is added."
        
        return {
            "summary": f"The model shows {bias_level} (severity: {severity}). Disparate Impact Ratio is {di} — values below 0.8 indicate legally significant bias. Statistical Parity Difference of {abs(spd):.3f} shows unequal prediction rates across groups in '{sensitive_col}'.",
            "disadvantaged_groups": f"Groups with lower selection rates or accuracy based on '{sensitive_col}' are most affected. Check the 'Accuracy by Group' chart for specifics.",
            "root_cause": f"Historical patterns in the dataset likely encode past discrimination. When the model learns from this data, it perpetuates those patterns. The '{sensitive_col}' column directly or indirectly influences the '{target_col}' predictions.",
            "recommendation": action,
            "gemini_powered": False
        }
    
    prompt = f"""
You are an AI fairness expert. Analyze these bias audit results and respond in JSON format only (no markdown, no backticks).

Dataset: {dataset_name}
Sensitive attribute (protected group): {sensitive_col}
Target variable (what model predicts): {target_col}

Fairness Metrics:
- Disparate Impact Ratio: {metrics['disparate_impact_ratio']} (threshold: 0.8 minimum, 1.0 = perfect)
- Statistical Parity Difference: {metrics['statistical_parity_difference']} (threshold: ±0.1, 0 = perfect)
- Equal Opportunity Difference: {metrics['equal_opportunity_difference']}
- Overall Model Accuracy: {metrics['overall_accuracy']}
- Per-group metrics: {json.dumps(metrics['group_metrics'], indent=2)}

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
            "summary": f"Analysis complete. Disparate Impact: {metrics['disparate_impact_ratio']}, Statistical Parity Diff: {metrics['statistical_parity_difference']:.4f}.",
            "disadvantaged_groups": "See group metrics chart for per-group breakdown.",
            "root_cause": "Could not generate detailed analysis. Check your Gemini API key.",
            "recommendation": "Apply fairness-aware training or data reweighing techniques.",
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
    classifier: str = Query("logistic", description="Classifier: logistic, random_forest, decision_tree"),
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
            "suggestions": get_mitigation_suggestion(metrics, severity),
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


@app.get("/health")
def health():
    return {"status": "healthy", "gemini": bool(GEMINI_API_KEY)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
