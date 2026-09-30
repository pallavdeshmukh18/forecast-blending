import React from 'react';
import { Layers } from 'lucide-react';

export default function ModelAgreement({ weightsData }) {
  const weights = weightsData?.weights || [
    { name: 'ECMWF IFS', percent: 34, color: '#0284c7' },
    { name: 'GraphCast', percent: 24, color: '#38bdf8' },
    { name: 'Pangu-Weather', percent: 20, color: '#818cf8' },
    { name: 'NOAA GFS', percent: 14, color: '#f59e0b' },
    { name: 'Regional WRF', percent: 8, color: '#10b981' },
  ];

  const agreementStatus = weightsData?.agreementStatus || 'Moderate model agreement';

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

      <div className="agreement-footer-status">
        <span className="status-indicator-dot"></span>
        <span className="agreement-status-text">{agreementStatus}</span>
      </div>
    </div>
  );
}
