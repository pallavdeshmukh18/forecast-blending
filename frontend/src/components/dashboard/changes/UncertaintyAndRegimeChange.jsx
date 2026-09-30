import React from 'react';
import { Layers, Wind, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';

export default function UncertaintyAndRegimeChange({ uncertaintyChange, regimeChange }) {
  const u = uncertaintyChange || {};
  const r = regimeChange || {};

  const isWidened = u.trend === 'widened';

  return (
    <div className="overview-card uncertainty-regime-change-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">DISPERSION & REGIME ANALYSIS</span>
          <h3 className="overview-card-title">UNCERTAINTY & WEATHER REGIME SHIFT</h3>
        </div>
        <div className="combo-icon-pill">
          <Layers size={16} color="#0284c7" />
        </div>
      </div>

      <div className="two-column-inner-grid">
        {/* Left: Uncertainty Trend */}
        <div className="uncertainty-trend-cell">
          <div className="trend-cell-top">
            <span className="sub-tag">UNCERTAINTY ENVELOPE</span>
            <span className={`status-badge ${isWidened ? 'badge-amber' : 'badge-green'}`}>
              {u.status || 'Similar Uncertainty'}
            </span>
          </div>

          <div className="spread-evolution-cluster">
            <div className="spread-block">
              <span className="sp-lbl">Previous Spread:</span>
              <span className="sp-val">{u.prevSpread} {u.unit}</span>
            </div>
            <ArrowRight size={16} className="spread-arrow" />
            <div className="spread-block">
              <span className="sp-lbl">Current Spread:</span>
              <span className="sp-val highlight-val">{u.currSpread} {u.unit}</span>
            </div>
          </div>

          <p className="trend-explainer">{u.description}</p>
        </div>

        {/* Right: Weather Regime Shift */}
        <div className="regime-shift-cell">
          <div className="trend-cell-top">
            <span className="sub-tag">WEATHER REGIME TRANSITION</span>
            <span className={`status-badge ${r.hasRegimeShifted ? 'badge-blue' : 'badge-neutral'}`}>
              {r.hasRegimeShifted ? 'Regime Shift Detected' : 'Weather Regime Unchanged'}
            </span>
          </div>

          <div className="regime-transition-display">
            <div className="regime-badge-item">
              <span className="r-stage-lbl">Previous Run</span>
              <span className="r-name-tag">{r.previousRegime}</span>
            </div>
            <ArrowRight size={18} className="regime-arrow" />
            <div className="regime-badge-item current-regime">
              <span className="r-stage-lbl">Current Run</span>
              <span className="r-name-tag active-regime">{r.currentRegime}</span>
            </div>
          </div>

          <p className="trend-explainer">{r.description}</p>
        </div>
      </div>
    </div>
  );
}
