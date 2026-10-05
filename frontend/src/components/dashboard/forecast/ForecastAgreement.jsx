import React from 'react';
import { Layers, HelpCircle, CheckCircle2 } from 'lucide-react';

export default function ForecastAgreement({ forecastData }) {
  const spread = forecastData?.spread || {
    min: 37,
    max: 51,
    spreadValue: 14,
    agreement: 'Moderate',
    confidence: '78%',
  };

  const unit = forecastData?.unit ?? 'mm';
  const blendVal = forecastData?.blendedValue ?? 42;

  // Calculate position along min-max range for the visual indicator
  const rangeSpan = Math.max(0.1, (spread.max - spread.min));
  const blendPercent = Math.min(95, Math.max(5, ((blendVal - spread.min) / rangeSpan) * 100));

  const getAgreementBadgeColor = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'high':
        return { text: '#059669', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.3)' };
      case 'moderate':
        return { text: '#d97706', bg: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.3)' };
      case 'low':
        return { text: '#dc2626', bg: 'rgba(239, 68, 68, 0.12)', border: 'rgba(239, 68, 68, 0.3)' };
      default:
        return { text: '#94a3b8', bg: 'rgba(148, 163, 184, 0.12)', border: 'rgba(148, 163, 184, 0.3)' };
    }
  };

  const agreementColors = getAgreementBadgeColor(spread.agreement);

  return (
    <div className="overview-card forecast-agreement-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">CONSENSUS METRICS</span>
          <h3 className="overview-card-title">MODEL AGREEMENT</h3>
        </div>
        <div className="agreement-top-icon">
          <Layers size={16} color="#3b82f6" />
        </div>
      </div>

      <div className="agreement-metrics-grid">
        {/* Model Range */}
        <div className="agreement-stat-box">
          <span className="agreement-stat-label">Model range</span>
          <span className="agreement-stat-value">
            {spread.min} {unit} — {spread.max} {unit}
          </span>
        </div>

        {/* Spread */}
        <div className="agreement-stat-box">
          <span className="agreement-stat-label">Spread</span>
          <span className="agreement-stat-value">
            {spread.spreadValue} {unit}
          </span>
        </div>

        {/* Agreement Status */}
        <div className="agreement-stat-box">
          <span className="agreement-stat-label">Agreement</span>
          <span
            className="agreement-status-pill"
            style={{
              color: agreementColors.text,
              backgroundColor: agreementColors.bg,
              borderColor: agreementColors.border,
            }}
          >
            {spread.agreement}
          </span>
        </div>

        {/* Confidence */}
        <div className="agreement-stat-box">
          <span className="agreement-stat-label">Confidence</span>
          <span className="agreement-stat-value highlight">
            {spread.confidence || 'Confidence data unavailable'}
          </span>
        </div>
      </div>

      {/* Visual Range Indicator Bar */}
      <div className="visual-range-container">
        <div className="range-track-labels">
          <span className="range-bound-label">
            Min: <strong>{spread.min}</strong>
          </span>
          <span className="range-center-title">Inter-Model Distribution</span>
          <span className="range-bound-label">
            Max: <strong>{spread.max}</strong>
          </span>
        </div>

        <div className="range-bar-track">
          {/* Active range interval */}
          <div className="range-bar-gradient"></div>

          {/* BharatWeather Blend marker */}
          <div
            className="range-marker blend-marker"
            style={{ left: `${blendPercent}%` }}
            title={`BharatWeather Blend: ${blendVal} ${unit}`}
          >
            <div className="marker-pin"></div>
            <div className="marker-tooltip">Blend {blendVal}</div>
          </div>

          {/* High Model marker */}
          <div
            className="range-marker max-marker"
            style={{ left: '96%' }}
            title={`Max Model: ${spread.max} ${unit}`}
          >
            <div className="marker-pin high-pin"></div>
            <div className="marker-tooltip">High {spread.max}</div>
          </div>
        </div>

        <div className="range-legend">
          <span className="legend-item">
            <span className="legend-dot blend-dot"></span> Blend Value ({blendVal} {unit})
          </span>
          <span className="legend-item">
            <span className="legend-dot spread-dot"></span> Total Spread ({spread.spreadValue} {unit})
          </span>
        </div>
      </div>
    </div>
  );
}
