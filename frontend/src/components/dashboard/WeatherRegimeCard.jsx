import React from 'react';
import { Compass, CheckCircle2 } from 'lucide-react';

export default function WeatherRegimeCard({ regimeData }) {
  const regimeName = regimeData?.name ?? 'Heavy Convective Monsoon';
  const confidence = regimeData?.confidence ?? '82%';
  const description =
    regimeData?.description ??
    'Current atmospheric conditions indicate a heavy-convective regime across the regional domain.';

  return (
    <div className="overview-card weather-regime-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">ATMOSPHERIC STATE</span>
          <h3 className="overview-card-title">ACTIVE SYNOPTIC REGIME</h3>
        </div>

        <div className="regime-status-dot-pill">
          <span className="live-pulsing-dot"></span>
          <span>Detected</span>
        </div>
      </div>

      <div className="regime-card-content">
        <div className="regime-badge-banner">
          <div className="regime-badge-icon">
            <Compass size={18} strokeWidth={2} />
          </div>
          <div className="regime-title-wrap">
            <h4 className="regime-primary-name">{regimeName}</h4>
            <div className="regime-confidence-row">
              <span className="regime-conf-label">Confidence:</span>
              <span className="regime-conf-val">{confidence}</span>
            </div>
          </div>
        </div>

        <p className="regime-explainer-text">{description}</p>

        <div className="regime-footer-features">
          <div className="regime-pill-indicator">
            <CheckCircle2 size={13} color="#059669" />
            <span>High Moisture Advection</span>
          </div>
          <div className="regime-pill-indicator">
            <CheckCircle2 size={13} color="#059669" />
            <span>Diurnal Peak Heating</span>
          </div>
        </div>
      </div>
    </div>
  );
}
