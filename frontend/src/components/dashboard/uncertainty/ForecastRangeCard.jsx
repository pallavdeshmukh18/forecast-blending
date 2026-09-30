import React from 'react';
import { Target, HelpCircle, Binary, ShieldCheck } from 'lucide-react';

export default function ForecastRangeCard({ data }) {
  const central = data?.centralForecast ?? 42.0;
  const unit = data?.unit || 'mm';
  const varName = data?.varName || 'Rainfall';
  const [low80, high80] = data?.predictionInterval80 || [32.0, 54.0];
  const [low90, high90] = data?.predictionInterval90 || [26.0, 62.0];
  const method = data?.calibrationMethod || 'Zero-Inflated Gamma EMOS';

  // Calculate percentage positions for range slider visualization
  const spanMin = Math.max(0, low90 * 0.85);
  const spanMax = high90 * 1.15;
  const totalSpan = spanMax - spanMin || 1;
  const left80Pct = Math.max(0, Math.min(100, ((low80 - spanMin) / totalSpan) * 100));
  const right80Pct = Math.max(0, Math.min(100, ((high80 - spanMin) / totalSpan) * 100));
  const centralPct = Math.max(0, Math.min(100, ((central - spanMin) / totalSpan) * 100));

  return (
    <div className="overview-card forecast-range-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">CALIBRATED PREDICTION ENVELOPE</span>
          <h3 className="overview-card-title">FORECAST RANGE & CONFIDENCE INTERVAL</h3>
        </div>
        <div className="range-icon-pill">
          <Target size={16} color="#0284c7" />
        </div>
      </div>

      <div className="forecast-range-body">
        {/* Left Column: Big Central Forecast */}
        <div className="range-central-column">
          <span className="range-stat-label">CENTRAL BLENDED FORECAST</span>
          <div className="range-central-val-cluster">
            <span className="range-central-number">{central}</span>
            <span className="range-central-unit">{unit}</span>
          </div>
          <span className="range-stat-sub">
            Calibrated ensemble median / mode
          </span>
        </div>

        {/* Right Column: Prediction Interval Range */}
        <div className="range-interval-column">
          <div className="interval-primary-row">
            <span className="interval-tier-badge">80% Prediction Interval</span>
            <div className="interval-bounds-text">
              <span className="bound-val">{low80} {unit}</span>
              <span className="bound-sep">—</span>
              <span className="bound-val">{high80} {unit}</span>
            </div>
          </div>

          {/* Visual Range Track */}
          <div className="range-visual-track-container">
            <div className="range-track-base">
              {/* 80% interval shaded bar */}
              <div
                className="range-interval-fill"
                style={{
                  left: `${left80Pct}%`,
                  width: `${right80Pct - left80Pct}%`,
                }}
              ></div>
              {/* Central forecast point */}
              <div
                className="range-central-pin"
                style={{ left: `${centralPct}%` }}
              >
                <div className="pin-head"></div>
              </div>
            </div>
            <div className="range-track-labels">
              <span>{low90} {unit} (P10)</span>
              <span className="label-center">Central: {central} {unit}</span>
              <span>{high90} {unit} (P90)</span>
            </div>
          </div>

          <div className="range-secondary-note">
            <span className="p90-chip">90% Bound: [{low90} — {high90} {unit}]</span>
            <span className="method-chip">{method}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
