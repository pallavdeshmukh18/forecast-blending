import React from 'react';
import { Flame, Clock, Maximize2, Zap } from 'lucide-react';

export default function ForecastSeverityCard({ selectedHazard }) {
  const forecastVal = selectedHazard?.forecastValue ?? 42;
  const peakVal = selectedHazard?.peakValue ?? 58;
  const unit = selectedHazard?.unit || 'mm';
  const duration = selectedHazard?.duration || '18 hours';
  const severity = selectedHazard?.severity || 'Warning';

  const getSeverityStyle = (sev) => {
    const s = (sev || '').toLowerCase();
    if (s.includes('severe') || s.includes('warning')) {
      return { text: '#b91c1c', bg: '#fef2f2', border: '#fecaca' };
    }
    if (s.includes('watch') || s.includes('moderate')) {
      return { text: '#c2410c', bg: '#fff7ed', border: '#ffedd5' };
    }
    return { text: '#0369a1', bg: '#f0f9ff', border: '#bae6fd' };
  };

  const badgeStyle = getSeverityStyle(severity);

  return (
    <div className="overview-card forecast-severity-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">MAGNITUDE & DURATION</span>
          <h3 className="overview-card-title">FORECAST SEVERITY</h3>
        </div>
        <div
          className="severity-pill-badge"
          style={{
            color: badgeStyle.text,
            background: badgeStyle.bg,
            borderColor: badgeStyle.border,
          }}
        >
          <Zap size={13} strokeWidth={2.5} />
          <span>{severity}</span>
        </div>
      </div>

      <div className="severity-grid-stats">
        {/* Expected Composite Value */}
        <div className="severity-stat-cell">
          <span className="sev-stat-label">Expected Value</span>
          <div className="sev-stat-cluster">
            <span className="sev-stat-val">{forecastVal}</span>
            <span className="sev-stat-unit">{unit}</span>
          </div>
          <span className="sev-stat-sub">Calibrated ensemble blend</span>
        </div>

        {/* Peak/Max Magnitude */}
        <div className="severity-stat-cell">
          <span className="sev-stat-label">Peak Maximum</span>
          <div className="sev-stat-cluster">
            <span className="sev-stat-val peak-color">{peakVal}</span>
            <span className="sev-stat-unit">{unit}</span>
          </div>
          <span className="sev-stat-sub">Worst-case member tail</span>
        </div>

        {/* Event Duration */}
        <div className="severity-stat-cell full-span">
          <span className="sev-stat-label">Expected Duration</span>
          <div className="sev-stat-cluster-inline">
            <Clock size={16} className="duration-icon" />
            <span className="sev-stat-duration-val">{duration}</span>
          </div>
          <span className="sev-stat-sub">Synoptic persistence window</span>
        </div>
      </div>
    </div>
  );
}
