import React from 'react';
import { Layers } from 'lucide-react';

const LEVEL_COLORS = { low: '#059669', moderate: '#d97706', high: '#dc2626', unavailable: '#64748b' };

// Skill-correlation reference from the audited WeatherBench 2 benchmark (README, "Error Diagnostics via Model Disagreement")
const SKILL_CORRELATION = { rainfall: 0.531, temperature: 0.45, wind_speed: null };

export default function ModelAgreement({ weightsData, disagreement = null }) {
  const weights = weightsData?.weights || [
    { name: 'ECMWF IFS', percent: 34, color: '#0284c7' },
    { name: 'GraphCast', percent: 24, color: '#38bdf8' },
    { name: 'Pangu-Weather', percent: 20, color: '#818cf8' },
    { name: 'NOAA GFS', percent: 14, color: '#f59e0b' },
    { name: 'Regional WRF', percent: 8, color: '#10b981' },
  ];

  const agreementStatus = weightsData?.agreementStatus || 'Moderate model agreement';
  const rows = disagreement?.disagreement || [];

  return (
    <div className="overview-card model-agreement-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">DYNAMIC FUSION</span>
          <h3 className="overview-card-title">MODEL AGREEMENT</h3>
        </div>
        <div className="agreement-icon-wrap">
          <Layers size={16} color="#2563eb" />
        </div>
      </div>

      <div className="model-weights-bars">
        {weights.map((item, idx) => (
          <div key={idx} className="agreement-bar-row">
            <div className="agreement-bar-labels">
              <span className="agreement-model-label">{item.name}</span>
              <span className="agreement-model-percent">{item.percent}%</span>
            </div>
            <div className="agreement-progress-track">
              <div
                className="agreement-progress-fill"
                style={{
                  width: `${item.percent}%`,
                  background: item.color,
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>
      {weightsData?.source && (
        <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 4 }}>{weightsData.source}</div>
      )}

      {rows.length > 0 && (
        <div style={{ marginTop: 12, borderTop: '1px solid rgba(148,163,184,.25)', paddingTop: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '.04em', marginBottom: 6 }}>
            LIVE DISAGREEMENT · {(disagreement.models_available || []).join(' vs ').replaceAll('_', ' ')}
          </div>
          {rows.map((r) => {
            const level = r.disagreement_level;
            const corr = SKILL_CORRELATION[r.variable];
            return (
              <div key={r.variable} title={r.interpretation} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, padding: '2px 0' }}>
                <span style={{ width: 84, textTransform: 'capitalize' }}>{r.variable.replace('_', ' ')}</span>
                <span style={{ fontVariantNumeric: 'tabular-nums', width: 54 }}>
                  {level === 'unavailable' ? '—' : `±${r.mean_disagreement}`}
                </span>
                <span style={{ color: LEVEL_COLORS[level], fontWeight: 700, textTransform: 'uppercase', fontSize: 10.5 }}>{level}</span>
                {corr != null && (
                  <span style={{ marginLeft: 'auto', color: '#64748b', fontSize: 10.5 }} title="Correlation of |HRES − GraphCast| disagreement with forecast error, WeatherBench 2 benchmark">
                    r = {corr.toFixed(2)} vs skill
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="agreement-footer-status">
        <span className="status-indicator-dot"></span>
        <span className="agreement-status-text">{agreementStatus}</span>
      </div>
    </div>
  );
}
