import React from 'react';
import { Target, CheckCircle2, SlidersHorizontal } from 'lucide-react';

export default function EventThresholdCard({
  thresholds,
  currentThresholdValue,
  onSelectThreshold,
  hazardName,
  unit,
}) {
  return (
    <div className="overview-card event-threshold-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">TRIGGER CRITERIA</span>
          <h3 className="overview-card-title">EVENT THRESHOLD</h3>
        </div>
        <div className="threshold-icon-pill">
          <Target size={16} color="#0284c7" />
        </div>
      </div>

      <div className="threshold-selector-list">
        {thresholds && thresholds.length > 0 ? (
          thresholds.map((t) => {
            const isSelected = Number(currentThresholdValue) === Number(t.value);
            return (
              <button
                type="button"
                key={t.value}
                className={`threshold-option-btn ${isSelected ? 'is-selected' : ''}`}
                onClick={() => onSelectThreshold && onSelectThreshold(t.value)}
              >
                <div className="thresh-btn-left">
                  <div className={`thresh-radio ${isSelected ? 'is-active' : ''}`}>
                    {isSelected && <span className="radio-inner-dot"></span>}
                  </div>
                  <span className="thresh-label-text">{t.label}</span>
                </div>
                {isSelected && (
                  <span className="thresh-active-tag">Active Filter</span>
                )}
              </button>
            );
          })
        ) : (
          <div className="threshold-option-btn is-selected">
            <span className="thresh-label-text">≥ 45 km/h (Gale Threshold)</span>
            <span className="thresh-active-tag">Standard Operational Threshold</span>
          </div>
        )}
      </div>

      <div className="threshold-explainer-box">
        <SlidersHorizontal size={13} className="explainer-icon" />
        <span>
          Exceedance probabilities are calibrated against meteorological verification standards for {hazardName.toLowerCase()}.
        </span>
      </div>
    </div>
  );
}
