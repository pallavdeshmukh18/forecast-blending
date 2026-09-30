import React from 'react';
import { Sliders, GitCommit, ShieldCheck, Binary } from 'lucide-react';

export default function CalibrationUncertaintyCard({ selectedHazard }) {
  const method = selectedHazard?.calibrationMethod || 'Zero-Inflated Gamma EMOS';
  const interval = selectedHazard?.predictionInterval || '[32 mm — 54 mm] (80%)';
  const upperTail = selectedHazard?.upperTailEstimate || '64 mm (P95 GEV)';
  const spread = selectedHazard?.spreadValue || '14 mm';

  return (
    <div className="overview-card calibration-uncertainty-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">POST-PROCESSING</span>
          <h3 className="overview-card-title">UNCERTAINTY & CALIBRATION</h3>
        </div>
        <div className="calibration-icon-pill">
          <Binary size={16} color="#0284c7" />
        </div>
      </div>

      <div className="calibration-specs-list">
        <div className="cal-spec-row">
          <span className="cal-spec-label">
            <Sliders size={13} className="cal-spec-icon" />
            <span>Calibration Framework</span>
          </span>
          <span className="cal-spec-val highlight-method">{method}</span>
        </div>

        <div className="cal-spec-row">
          <span className="cal-spec-label">
            <GitCommit size={13} className="cal-spec-icon" />
            <span>80% Prediction Interval</span>
          </span>
          <span className="cal-spec-val monospace">{interval}</span>
        </div>

        <div className="cal-spec-row">
          <span className="cal-spec-label">
            <ShieldCheck size={13} className="cal-spec-icon" />
            <span>Upper-Tail Estimate</span>
          </span>
          <span className="cal-spec-val highlight-tail monospace">{upperTail}</span>
        </div>

        <div className="cal-spec-row">
          <span className="cal-spec-label">
            <GitCommit size={13} className="cal-spec-icon" />
            <span>Ensemble Spread (Std Dev)</span>
          </span>
          <span className="cal-spec-val monospace">{spread}</span>
        </div>
      </div>
    </div>
  );
}
