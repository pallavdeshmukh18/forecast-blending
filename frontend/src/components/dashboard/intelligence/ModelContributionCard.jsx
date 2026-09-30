import React from 'react';
import { PieChart, Cpu } from 'lucide-react';

export default function ModelContributionCard({ contributions }) {
  const models = contributions || [
    { id: 'ecmwf', name: 'ECMWF IFS', percent: 34, color: '#0284c7' },
    { id: 'graphcast', name: 'GraphCast', percent: 24, color: '#38bdf8' },
    { id: 'pangu', name: 'Pangu-Weather', percent: 20, color: '#818cf8' },
    { id: 'gfs', name: 'NOAA GFS', percent: 14, color: '#f59e0b' },
    { id: 'regional_wrf', name: 'Regional WRF', percent: 8, color: '#10b981' },
  ];

  return (
    <div className="overview-card model-contribution-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">CONVEX WEIGHTS</span>
          <h3 className="overview-card-title">MODEL CONTRIBUTION</h3>
        </div>
        <div className="contribution-header-icon">
          <PieChart size={16} color="#0284c7" />
        </div>
      </div>

      <div className="contribution-bars-list">
        {models.map((item) => (
          <div key={item.id} className="contribution-bar-row">
            <div className="contribution-meta">
              <span className="contribution-name">{item.name}</span>
              <span className="contribution-percent">{item.percent}%</span>
            </div>
            <div className="contribution-track">
              <div
                className="contribution-fill"
                style={{
                  width: `${item.percent}%`,
                  backgroundColor: item.color || '#3b82f6',
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      <div className="contribution-footer">
        <span className="contribution-sum-label">Total Convex Weight Sum:</span>
        <span className="contribution-sum-val">100% (Constrained)</span>
      </div>
    </div>
  );
}
