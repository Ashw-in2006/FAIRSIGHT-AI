import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell, Legend
} from 'recharts';
import toast, { Toaster } from 'react-hot-toast';

const API = 'http://localhost:8000';

// ─── Color helpers ─────────────────────────────────────────────────────────
const SEV_COLOR = { Critical: '#fc8181', High: '#f6ad55', Medium: '#f6e05e', Low: '#68d391' };
const SEV_BG = {
  Critical: 'rgba(252,129,129,0.1)',
  High: 'rgba(246,173,85,0.1)',
  Medium: 'rgba(246,224,94,0.1)',
  Low: 'rgba(104,211,145,0.1)'
};

const BAR_COLORS = ['#63b3ed', '#4fd1c5', '#b794f4', '#f6ad55', '#fc8181', '#68d391', '#f6e05e', '#ed8936'];

// ─── Sub-components ────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, color, mono }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid var(--border)`,
      borderRadius: 'var(--radius)',
      padding: '1.1rem 1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
      transition: 'border-color 0.2s',
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border2)'}
      onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
    >
      <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
      <span style={{ fontSize: 26, fontWeight: 700, fontFamily: mono ? 'var(--font-mono)' : 'var(--font-display)', color: color || 'var(--accent)', lineHeight: 1.1 }}>{value}</span>
      {sub && <span style={{ fontSize: 11, color: 'var(--text3)' }}>{sub}</span>}
    </div>
  );
}

function SeverityBadge({ severity }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 12px', borderRadius: 999,
      background: SEV_BG[severity],
      border: `1px solid ${SEV_COLOR[severity]}44`,
      color: SEV_COLOR[severity],
      fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)',
      letterSpacing: '0.05em',
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: SEV_COLOR[severity], display: 'inline-block' }} />
      {severity}
    </span>
  );
}

function Section({ title, children, badge }) {
  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '1rem' }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.01em' }}>{title}</h2>
        {badge}
        <div style={{ flex: 1, height: 1, background: 'var(--border)', marginLeft: 8 }} />
      </div>
      {children}
    </div>
  );
}

function Pill({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 12px',
      background: 'var(--bg3)',
      borderRadius: 'var(--radius-sm)',
      border: '1px solid var(--border)',
      fontSize: 13,
    }}>
      <span style={{ color: 'var(--text2)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: color || 'var(--accent)', fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function SuggestionList({ suggestions }) {
  if (!suggestions?.length) return null;

  return (
    <Section title="How to Fix This" badge={<span style={{ fontSize: 11, color: 'var(--accent)', fontFamily: 'var(--font-mono)', background: 'rgba(99,179,237,0.08)', padding: '2px 8px', borderRadius: 999 }}>Mitigation</span>}>
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem 1.1rem' }}>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {suggestions.map((s, i) => (
            <li key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start', color: 'var(--text)', fontSize: 13.5, lineHeight: 1.6 }}>
              <span style={{ color: 'var(--accent2)', flexShrink: 0 }}>•</span>
              <span>{s}</span>
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}

function ProblemPopup({ onClose }) {
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.7)', display: 'grid', placeItems: 'center', zIndex: 200 }} onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Bias audit guide"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(720px, calc(100vw - 24px))',
          background: 'var(--surface)',
          border: '1px solid var(--border2)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.25rem',
          boxShadow: '0 24px 90px rgba(0,0,0,0.45)'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 800, marginBottom: 6 }}>🛡️ Unbiased AI Decision</div>
            <p style={{ color: 'var(--text2)', lineHeight: 1.6 }}>
              Detect hidden unfairness in automated decisions before it affects hiring, loans, or healthcare.
            </p>
          </div>
          <button onClick={onClose} style={{ background: 'var(--bg3)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 999, width: 32, height: 32, cursor: 'pointer' }}>✕</button>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          {[
            ['⚖️ Fairness', 'Measure bias with disparate impact and parity differences.'],
            ['🔍 Detection', 'Flag hidden discrimination across protected groups.'],
            ['🔧 Fix', 'Suggest reweighing, threshold tuning, and data improvements.'],
            ['📈 Explain', 'Show clear charts and plain-English guidance.'],
          ].map(([icon, text]) => (
            <div key={text} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '0.85rem 1rem', color: 'var(--text2)' }}>
              <strong style={{ color: 'var(--text)', marginRight: 8 }}>{icon}</strong>
              {text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const customTooltipStyle = {
  background: '#1a2332',
  border: '1px solid rgba(99,179,237,0.2)',
  borderRadius: 8,
  fontSize: 12,
  fontFamily: 'var(--font-mono)',
  color: '#e2e8f0',
};

// ─── Column Selector Step ─────────────────────────────────────────────────

function ColumnSelector({ preview, onAudit, loading }) {
  const [targetCol, setTargetCol] = useState('');
  const [sensitiveCol, setSensitiveCol] = useState('');
  const [classifier, setClassifier] = useState('logistic');
  const [fileToSend, setFileToSend] = useState(preview.file);

  const columns = preview.columns;
  const canRun = targetCol && sensitiveCol && targetCol !== sensitiveCol;

  const selectStyle = {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    padding: '10px 12px',
    color: 'var(--text)',
    fontSize: 13,
    fontFamily: 'var(--font-mono)',
    width: '100%',
    cursor: 'pointer',
    outline: 'none',
  };

  return (
    <div>
      <Section title="Dataset Preview">
        <div style={{
          overflowX: 'auto',
          borderRadius: 'var(--radius)',
          border: '1px solid var(--border)',
          marginBottom: '1rem',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
            <thead>
              <tr style={{ background: 'var(--surface)' }}>
                {columns.map(c => (
                  <th key={c.name} style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--accent)', borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap', fontWeight: 600 }}>
                    {c.name}
                    <div style={{ fontSize: 10, color: 'var(--text3)', fontWeight: 400 }}>{c.dtype}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.preview.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  {columns.map(c => (
                    <td key={c.name} style={{ padding: '7px 12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>
                      {String(row[c.name] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
            {preview.rows.toLocaleString()} rows · {columns.length} columns · {preview.filename}
          </span>
        </div>
      </Section>

      <Section title="Configure Audit">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 16 }}>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Target Column (what you predict)
            </label>
            <select style={selectStyle} value={targetCol} onChange={e => setTargetCol(e.target.value)}>
              <option value="">— Select —</option>
              {columns.map(c => <option key={c.name} value={c.name}>{c.name} ({c.unique_count} unique)</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Sensitive Column (protected attribute)
            </label>
            <select style={selectStyle} value={sensitiveCol} onChange={e => setSensitiveCol(e.target.value)}>
              <option value="">— Select —</option>
              {columns.map(c => <option key={c.name} value={c.name}>{c.name} ({c.unique_count} unique)</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Classifier Model
            </label>
            <select style={selectStyle} value={classifier} onChange={e => setClassifier(e.target.value)}>
              <option value="logistic">Logistic Regression</option>
              <option value="random_forest">Random Forest</option>
              <option value="decision_tree">Decision Tree</option>
            </select>
          </div>
        </div>

        {/* Column hints */}
        {columns.length > 0 && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
            {columns.map(c => (
              <button
                key={c.name}
                onClick={() => {
                  if (!targetCol) setTargetCol(c.name);
                  else if (!sensitiveCol && c.name !== targetCol) setSensitiveCol(c.name);
                }}
                style={{
                  background: c.name === targetCol ? 'rgba(99,179,237,0.15)' : c.name === sensitiveCol ? 'rgba(79,209,197,0.15)' : 'var(--bg3)',
                  border: `1px solid ${c.name === targetCol ? 'var(--accent)' : c.name === sensitiveCol ? 'var(--accent2)' : 'var(--border)'}`,
                  borderRadius: 999,
                  padding: '3px 10px',
                  fontSize: 11,
                  color: c.name === targetCol ? 'var(--accent)' : c.name === sensitiveCol ? 'var(--accent2)' : 'var(--text3)',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {c.name}
                {c.name === targetCol && ' ▸ target'}
                {c.name === sensitiveCol && ' ▸ sensitive'}
              </button>
            ))}
          </div>
        )}

        <button
          onClick={() => onAudit(preview.file, targetCol, sensitiveCol, classifier)}
          disabled={!canRun || loading}
          style={{
            background: canRun && !loading ? 'var(--accent)' : 'var(--surface)',
            color: canRun && !loading ? '#080c10' : 'var(--text3)',
            border: 'none',
            borderRadius: 'var(--radius)',
            padding: '12px 28px',
            fontFamily: 'var(--font-display)',
            fontWeight: 700,
            fontSize: 15,
            cursor: canRun && !loading ? 'pointer' : 'not-allowed',
            width: '100%',
            letterSpacing: '-0.01em',
          }}
        >
          {loading ? '⏳  Analyzing bias...' : '→  Run Bias Audit'}
        </button>
      </Section>
    </div>
  );
}

// ─── Results Panel ─────────────────────────────────────────────────────────

function Results({ data }) {
  const { severity, metrics, explanation, dataset_info, suggestions } = data;
  const gm = metrics.group_metrics;

  // Accuracy by group chart data
  const accData = Object.entries(gm.accuracy || {}).map(([group, val]) => ({
    group: String(group).length > 12 ? String(group).slice(0, 12) + '…' : String(group),
    accuracy: Math.round(val * 100),
  }));

  // Selection rate by group
  const selData = Object.entries(gm.selection_rate || {}).map(([group, val]) => ({
    group: String(group).length > 12 ? String(group).slice(0, 12) + '…' : String(group),
    rate: Math.round(val * 100),
  }));

  // False positive rate by group
  const fprData = Object.entries(gm.false_positive_rate || {}).map(([group, val]) => ({
    group: String(group).length > 12 ? String(group).slice(0, 12) + '…' : String(group),
    fpr: Math.round(val * 100),
  }));

  // Feature importance radar/bar
  const fiData = Object.entries(metrics.feature_importance || {}).map(([feat, imp]) => ({
    feature: feat.length > 14 ? feat.slice(0, 14) + '…' : feat,
    importance: Math.round(imp * 100),
  }));

  const fairnessScore = Math.round(
    (Math.min(metrics.disparate_impact_ratio, 1) * 40) +
    (Math.max(0, 1 - Math.abs(metrics.statistical_parity_difference) * 5) * 30) +
    (Math.max(0, 1 - Math.abs(metrics.equal_opportunity_difference) * 5) * 30)
  );

  return (
    <div>
      {/* Header */}
      <Section
        title="Audit Results"
        badge={<SeverityBadge severity={severity} />}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 10 }}>
          <MetricCard
            label="Disparate Impact"
            value={metrics.disparate_impact_ratio}
            sub="≥0.8 = acceptable"
            color={metrics.disparate_impact_ratio < 0.8 ? 'var(--red)' : metrics.disparate_impact_ratio < 0.9 ? 'var(--orange)' : 'var(--green)'}
            mono
          />
          <MetricCard
            label="Statistical Parity Δ"
            value={metrics.statistical_parity_difference.toFixed(3)}
            sub="0 = perfectly fair"
            color={Math.abs(metrics.statistical_parity_difference) > 0.1 ? 'var(--red)' : Math.abs(metrics.statistical_parity_difference) > 0.05 ? 'var(--orange)' : 'var(--green)'}
            mono
          />
          <MetricCard
            label="Equal Opportunity Δ"
            value={metrics.equal_opportunity_difference.toFixed(3)}
            sub="0 = perfectly fair"
            color={Math.abs(metrics.equal_opportunity_difference) > 0.1 ? 'var(--red)' : 'var(--orange)'}
            mono
          />
          <MetricCard
            label="Overall Accuracy"
            value={(metrics.overall_accuracy * 100).toFixed(1) + '%'}
            sub={`Model: ${metrics.model_used}`}
            color="var(--accent)"
            mono
          />
        </div>

        {/* Fairness Score */}
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: 16,
        }}>
          <div style={{ flexShrink: 0 }}>
            <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Fairness Score</div>
            <div style={{ fontSize: 36, fontWeight: 800, fontFamily: 'var(--font-display)', color: fairnessScore >= 70 ? 'var(--green)' : fairnessScore >= 50 ? 'var(--orange)' : 'var(--red)', lineHeight: 1 }}>
              {fairnessScore}<span style={{ fontSize: 16, color: 'var(--text3)' }}>/100</span>
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ background: 'var(--bg3)', borderRadius: 999, height: 8, overflow: 'hidden' }}>
              <div style={{
                width: fairnessScore + '%',
                height: '100%',
                borderRadius: 999,
                background: fairnessScore >= 70 ? 'var(--green)' : fairnessScore >= 50 ? 'var(--orange)' : 'var(--red)',
                transition: 'width 1s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <span style={{ fontSize: 10, color: 'var(--text3)' }}>0 — Critical</span>
              <span style={{ fontSize: 10, color: 'var(--text3)' }}>100 — Perfect</span>
            </div>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>Train / Test</div>
            <div style={{ fontSize: 14, fontFamily: 'var(--font-mono)', color: 'var(--text2)' }}>
              {metrics.train_size} / {metrics.test_size}
            </div>
          </div>
        </div>
      </Section>

      {/* Charts Row */}
      <Section title="Group Metrics">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Accuracy by group */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem' }}>
            <div style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--font-mono)', marginBottom: 12 }}>Accuracy by {dataset_info.sensitive_col}</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={accData} barCategoryGap="35%">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.08)" />
                <XAxis dataKey="group" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                <Tooltip contentStyle={customTooltipStyle} formatter={v => [v + '%', 'Accuracy']} />
                <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                  {accData.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Selection rate by group */}
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem' }}>
            <div style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--font-mono)', marginBottom: 12 }}>Selection Rate by {dataset_info.sensitive_col}</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={selData} barCategoryGap="35%">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.08)" />
                <XAxis dataKey="group" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                <Tooltip contentStyle={customTooltipStyle} formatter={v => [v + '%', 'Selection Rate']} />
                <Bar dataKey="rate" radius={[4, 4, 0, 0]}>
                  {selData.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 2) % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* False Positive Rate */}
        {fprData.length > 0 && (
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem', marginTop: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--font-mono)', marginBottom: 12 }}>False Positive Rate by {dataset_info.sensitive_col}</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={fprData} barCategoryGap="35%">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.08)" />
                <XAxis dataKey="group" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                <Tooltip contentStyle={customTooltipStyle} formatter={v => [v + '%', 'FPR']} />
                <Bar dataKey="fpr" radius={[4, 4, 0, 0]}>
                  {fprData.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 4) % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* Feature Importance */}
      {fiData.length > 0 && (
        <Section title="Feature Importance (Top Predictors)">
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '1rem' }}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={fiData} layout="vertical" barCategoryGap="25%">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,0.08)" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} width={110} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {fiData.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}

      {/* Gemini Explanation */}
      <Section
        title="AI Analysis"
        badge={
          explanation.gemini_powered
            ? <span style={{ fontSize: 11, color: 'var(--accent2)', fontFamily: 'var(--font-mono)', background: 'rgba(79,209,197,0.1)', padding: '2px 8px', borderRadius: 999, border: '1px solid rgba(79,209,197,0.2)' }}>Gemini</span>
            : <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', background: 'var(--bg3)', padding: '2px 8px', borderRadius: 999 }}>Local</span>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[
            { key: 'summary', label: '📊 Verdict', color: SEV_COLOR[severity] },
            { key: 'disadvantaged_groups', label: '👥 Affected Groups', color: 'var(--purple)' },
            { key: 'root_cause', label: '🔍 Root Cause', color: 'var(--orange)' },
            { key: 'recommendation', label: '✅ Recommendation', color: 'var(--green)' },
          ].map(({ key, label, color }) => (
            <div key={key} style={{
              background: 'var(--bg3)',
              border: '1px solid var(--border)',
              borderLeft: `3px solid ${color}`,
              borderRadius: 'var(--radius-sm)',
              padding: '12px 14px',
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: color, fontFamily: 'var(--font-mono)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
              <div style={{ fontSize: 13.5, color: 'var(--text)', lineHeight: 1.65 }}>{explanation[key]}</div>
            </div>
          ))}
        </div>
      </Section>

      <SuggestionList suggestions={suggestions} />

      {/* Dataset Info */}
      <Section title="Dataset Summary">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 8, textTransform: 'uppercase' }}>Target Distribution ({dataset_info.target_col})</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {Object.entries(dataset_info.target_distribution).map(([k, v]) => (
                <Pill key={k} label={String(k)} value={`${v.toLocaleString()} rows`} />
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginBottom: 8, textTransform: 'uppercase' }}>Sensitive Distribution ({dataset_info.sensitive_col})</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {Object.entries(dataset_info.sensitive_distribution).map(([k, v]) => (
                <Pill key={k} label={String(k)} value={`${v.toLocaleString()} rows`} color="var(--accent2)" />
              ))}
            </div>
          </div>
        </div>
      </Section>
    </div>
  );
}

// ─── Upload Zone ─────────────────────────────────────────────────────────

function UploadZone({ onPreview, loading }) {
  const onDrop = useCallback(async (files) => {
    if (!files[0]) return;
    const f = files[0];
    const formData = new FormData();
    formData.append('file', f);

    try {
      const res = await axios.post(API + '/preview', formData);
      res.data.file = f;
      onPreview(res.data);
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Could not read CSV');
    }
  }, [onPreview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    disabled: loading,
  });

  return (
    <div
      {...getRootProps()}
      style={{
        border: `2px dashed ${isDragActive ? 'var(--accent)' : 'var(--border2)'}`,
        borderRadius: 'var(--radius-lg)',
        padding: '3rem 2rem',
        textAlign: 'center',
        cursor: loading ? 'not-allowed' : 'pointer',
        background: isDragActive ? 'rgba(99,179,237,0.04)' : 'var(--surface)',
        transition: 'all 0.2s',
      }}
    >
      <input {...getInputProps()} />
      <div style={{ fontSize: 32, marginBottom: 12 }}>📂</div>
      <p style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 17, marginBottom: 6, color: isDragActive ? 'var(--accent)' : 'var(--text)' }}>
        {isDragActive ? 'Drop it!' : 'Drop your CSV here'}
      </p>
      <p style={{ fontSize: 13, color: 'var(--text3)' }}>or click to browse · accepts .csv files</p>
      <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)', background: 'var(--bg3)', display: 'inline-block', padding: '4px 12px', borderRadius: 999 }}>
        try: adult_with_headers.csv · target: income · sensitive: sex
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────

export default function App() {
  const [step, setStep] = useState('upload'); // upload | configure | results
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState(null);
  const [theme, setTheme] = useState('dark');
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    axios.get(API + '/').then(r => setApiStatus(r.data)).catch(() => setApiStatus(null));
  }, []);

  const handlePreview = (data) => {
    setPreview(data);
    setStep('configure');
    toast.success(`Loaded ${data.rows.toLocaleString()} rows`);
  };

  const handleAudit = async (file, targetCol, sensitiveCol, classifier) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    const toastId = toast.loading('Running bias audit…');

    try {
      const res = await axios.post(
        `${API}/audit?target_col=${encodeURIComponent(targetCol)}&sensitive_col=${encodeURIComponent(sensitiveCol)}&classifier=${classifier}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResults(res.data);
      setStep('results');
      toast.success('Audit complete!', { id: toastId });
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Audit failed', { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setStep('upload'); setPreview(null); setResults(null); };

  const steps = [
    { key: 'upload', label: '01 Upload' },
    { key: 'configure', label: '02 Configure' },
    { key: 'results', label: '03 Results' },
  ];

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <Toaster position="top-right" toastOptions={{
        style: { background: '#1a2332', color: '#e2e8f0', border: '1px solid rgba(99,179,237,0.2)', fontFamily: 'var(--font-mono)', fontSize: 13 }
      }} />

      {/* Header */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        padding: '0 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: 56,
        position: 'sticky',
        top: 0,
        background: 'rgba(8,12,16,0.9)',
        backdropFilter: 'blur(12px)',
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 18, color: 'var(--accent)', letterSpacing: '-0.02em' }}>FairSight</span>
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text3)', background: 'var(--bg3)', padding: '2px 7px', borderRadius: 999, border: '1px solid var(--border)' }}>AI · v2.0</span>
        </div>

        {/* Step indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {steps.map((s, i) => (
            <React.Fragment key={s.key}>
              <span style={{
                fontSize: 11, fontFamily: 'var(--font-mono)',
                color: step === s.key ? 'var(--accent)' : ['upload', 'configure', 'results'].indexOf(step) > i ? 'var(--text2)' : 'var(--text3)',
                fontWeight: step === s.key ? 600 : 400,
              }}>{s.label}</span>
              {i < 2 && <span style={{ color: 'var(--border2)', fontSize: 10 }}>›</span>}
            </React.Fragment>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '4px 10px', color: 'var(--text2)', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>
            {theme === 'dark' ? '☀ Light' : '🌙 Dark'}
          </button>
          <button onClick={() => setShowInfo(true)} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '4px 10px', color: 'var(--text2)', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>
            ❓ Bias Guide
          </button>
          {apiStatus && (
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--green)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} />
              API online {apiStatus.gemini_configured ? '· Gemini ✓' : '· Gemini ✗'}
            </span>
          )}
          {!apiStatus && (
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--red)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--red)', display: 'inline-block' }} />
              API offline
            </span>
          )}
          {step !== 'upload' && (
            <button onClick={reset} style={{
              background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              padding: '4px 10px', color: 'var(--text2)', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--font-mono)'
            }}>← New Audit</button>
          )}
        </div>
      </header>

      {showInfo && <ProblemPopup onClose={() => setShowInfo(false)} />}

      {/* Main */}
      <main style={{ maxWidth: 960, margin: '0 auto', padding: '2rem 1.5rem' }}>
        {step === 'upload' && (
          <div>
            {/* Hero */}
            <div style={{ marginBottom: '2.5rem' }}>
              <h1 style={{
                fontFamily: 'var(--font-display)', fontWeight: 800,
                fontSize: 'clamp(28px, 5vw, 44px)',
                lineHeight: 1.1, letterSpacing: '-0.03em',
                marginBottom: '0.75rem',
                background: 'linear-gradient(135deg, var(--text) 40%, var(--accent) 100%)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>
                Detect bias in your<br />ML models.
              </h1>
              <p style={{ color: 'var(--text2)', fontSize: 15, maxWidth: 480, lineHeight: 1.65 }}>
                Upload any CSV dataset, select your target and protected attribute, and get a full fairness audit powered by Fairlearn + Gemini AI.
              </p>
            </div>

            <UploadZone onPreview={handlePreview} loading={loading} />

            {/* Feature pills */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: '1.5rem' }}>
              {['Disparate Impact', 'Statistical Parity', 'Equal Opportunity', 'Per-group Accuracy', 'Gemini Explanations', 'Feature Importance'].map(f => (
                <span key={f} style={{
                  fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text3)',
                  padding: '3px 10px', borderRadius: 999, border: '1px solid var(--border)',
                  background: 'var(--bg3)',
                }}>✓ {f}</span>
              ))}
            </div>
          </div>
        )}

        {step === 'configure' && preview && (
          <ColumnSelector preview={preview} onAudit={handleAudit} loading={loading} />
        )}

        {step === 'results' && results && (
          <Results data={results} />
        )}
      </main>
    </div>
  );
}
