import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Clock, Target, Info } from 'lucide-react';

export default function ActiveHazardBanner({ selectedHazard, locationName, leadTime }) {
  const isHazardActive = selectedHazard?.isHazardActive ?? false;
  const hazardName = selectedHazard?.hazardName || 'HEAVY RAINFALL';
  const prob = selectedHazard?.probability ?? 72;
  const horizon = leadTime ? `+${String(leadTime).replace('+', '').replace('h', '')}h` : '+24h';
  const threshold = selectedHazard?.threshold || '≥ 50 mm';
  const forecastVal = selectedHazard?.forecastValue ?? 42;
  const unit = selectedHazard?.unit || 'mm';

  if (!isHazardActive) {
    return (
      <div className="no-hazard-banner">
        <div className="no-hazard-icon-box">
          <CheckCircle size={22} color="#059669" strokeWidth={2.2} />
        </div>
        <div className="no-hazard-content">
          <h3 className="no-hazard-title">No significant hazard detected for the selected configuration.</h3>
          <p className="no-hazard-sub">
            Current forecast for {locationName} at {horizon} is{' '}
            <strong>
              {forecastVal} {unit}
            </strong>{' '}
            (Threshold: {threshold} with calibrated exceedance probability of {prob}%).
          </p>
        </div>
        <div className="no-hazard-status-pill">
          <span className="live-status-dot status-green"></span>
          <span>Nominal Conditions</span>
        </div>
      </div>
    );
  }

  return (
    <div className="active-hazard-banner">
      <div className="active-hazard-glow-overlay"></div>

      <div className="hazard-banner-left">
        <div className="hazard-badge-alert">
          <AlertTriangle size={15} strokeWidth={2.5} />
          <span>ACTIVE HAZARD ALERT</span>
        </div>

        <h2 className="hazard-banner-title">{hazardName}</h2>
        <span className="hazard-banner-location">
          Station Target: <strong>{locationName}</strong>
        </span>
      </div>

      <div className="hazard-banner-metrics">
        {/* Probability Metric */}
        <div className="hazard-metric-card">
          <span className="metric-tag-label">CALIBRATED PROBABILITY</span>
          <div className="metric-huge-val">
            <span className="prob-number">{prob}%</span>
          </div>
          <span className="metric-sub-note">Exceedance probability</span>
        </div>

        {/* Forecast Horizon */}
        <div className="hazard-metric-card">
          <span className="metric-tag-label">FORECAST HORIZON</span>
          <div className="metric-horizon-val">
            <Clock size={18} className="clock-sub-icon" />
            <span className="horizon-text">{horizon}</span>
          </div>
          <span className="metric-sub-note">Target forecast step</span>
        </div>

        {/* Expected Threshold */}
        <div className="hazard-metric-card">
          <span className="metric-tag-label">EXPECTED THRESHOLD</span>
          <div className="metric-thresh-val">
            <Target size={18} className="target-sub-icon" />
            <span className="thresh-text">{threshold}</span>
          </div>
          <span className="metric-sub-note">Trigger boundary</span>
        </div>
      </div>
    </div>
  );
}
