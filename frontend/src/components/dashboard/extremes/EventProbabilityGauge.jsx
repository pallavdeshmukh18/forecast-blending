import React from 'react';
import { Gauge, Info } from 'lucide-react';

export default function EventProbabilityGauge({ probability, threshold, hazardName, calibrationMethod }) {
  const prob = typeof probability === 'number' ? probability : parseInt(String(probability), 10) || 72;

  // Semi-circle gauge geometry
  const radius = 72;
  const strokeWidth = 12;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = Math.PI * normalizedRadius; // Half circle length
  const strokeDashoffset = circumference - (prob / 100) * circumference;

  const getColor = (p) => {
    if (p >= 65) return '#dc2626';
    if (p >= 40) return '#ea580c';
    if (p >= 20) return '#eab308';
    return '#10b981';
  };

  const currentColor = getColor(prob);

  return (
    <div className="overview-card event-probability-gauge-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">CALIBRATED RISK ENGINE</span>
          <h3 className="overview-card-title">EVENT PROBABILITY</h3>
        </div>
        <div className="gauge-icon-pill">
          <Gauge size={16} color={currentColor} />
        </div>
      </div>

      <div className="gauge-visual-center">
        <svg
          width="200"
          height="120"
          viewBox="0 0 160 95"
          className="gauge-svg"
        >
          {/* Background Arc */}
          <path
            d="M 10 85 A 70 70 0 0 1 150 85"
            fill="none"
            stroke="rgba(226, 232, 240, 0.9)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Progress Arc */}
          <path
            d="M 10 85 A 70 70 0 0 1 150 85"
            fill="none"
            stroke={currentColor}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="gauge-arc-fill"
          />

          {/* Center Text */}
          <text
            x="80"
            y="72"
            textAnchor="middle"
            fill="#0f172a"
            fontSize="28px"
            fontWeight="900"
            className="gauge-text-val"
          >
            {prob}%
          </text>
        </svg>

        <span className="gauge-state-subtitle">
          Probability of exceeding selected event threshold
        </span>
      </div>

      <div className="gauge-footer-details">
        <div className="gauge-detail-item">
          <span className="g-label">Target Threshold:</span>
          <span className="g-val">{threshold}</span>
        </div>
        <div className="gauge-detail-item">
          <span className="g-label">Calibration Engine:</span>
          <span className="g-val highlight-engine">{calibrationMethod || 'EMOS / GEV'}</span>
        </div>
      </div>
    </div>
  );
}
