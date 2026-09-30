import React, { useState } from 'react';
import { TrendingUp, Clock, ShieldCheck } from 'lucide-react';

export default function SkillByLeadTime({ leadTimeHorizons, unit, varName }) {
  const [activeMetric, setActiveMetric] = useState('rmse'); // 'rmse' or 'mae'

  const horizons = leadTimeHorizons || [];
  
  // Find max error value to normalize bar widths
  let maxError = 1;
  horizons.forEach((h) => {
    (h.models || []).forEach((m) => {
      const val = activeMetric === 'rmse' ? m.rmse : m.mae;
      if (val > maxError) maxError = val;
    });
  });

  return (
    <div className="overview-card skill-by-leadtime-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">ERROR EVOLUTION</span>
          <h3 className="overview-card-title">SKILL BY FORECAST HORIZON</h3>
        </div>
        <div className="leadtime-metric-toggle">
          <button
            type="button"
            className={`toggle-btn ${activeMetric === 'rmse' ? 'active' : ''}`}
            onClick={() => setActiveMetric('rmse')}
          >
            RMSE
          </button>
          <button
            type="button"
            className={`toggle-btn ${activeMetric === 'mae' ? 'active' : ''}`}
            onClick={() => setActiveMetric('mae')}
          >
            MAE
          </button>
        </div>
      </div>

      <div className="horizons-visual-grid">
        {horizons.map((h) => (
          <div key={h.leadTime} className="horizon-column-card">
            <div className="horizon-col-header">
              <div className="col-horizon-badge">
                <Clock size={13} />
                <span>{h.leadTime}</span>
              </div>
              <span className="col-sub-note">Forecast Horizon</span>
            </div>

            <div className="horizon-models-list">
              {(h.models || []).map((m) => {
                const isBlend = !!m.isBlend;
                const val = activeMetric === 'rmse' ? m.rmse : m.mae;
                const pctWidth = Math.min(100, Math.round((val / (maxError * 1.05)) * 100));

                return (
                  <div
                    key={m.id}
                    className={`horizon-model-item ${isBlend ? 'is-blend-item' : ''}`}
                  >
                    <div className="h-item-top">
                      <div className="h-model-name-wrap">
                        {isBlend && <ShieldCheck size={13} className="h-blend-icon" />}
                        <span className={isBlend ? 'h-model-blend-name' : 'h-model-name'}>
                          {m.name}
                        </span>
                      </div>
                      <span className="h-item-val monospace">
                        <strong>{val.toFixed(2)}</strong> {unit}
                      </span>
                    </div>

                    <div className="h-bar-track">
                      <div
                        className={`h-bar-fill ${isBlend ? 'blend-bar-fill' : 'standard-bar-fill'}`}
                        style={{ width: `${pctWidth}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="leadtime-summary-explainer">
        <span>
          Verification errors naturally scale with lead time due to non-linear atmospheric chaos. The adaptive SkyCast blend demonstrates superior persistence in error suppression across extended horizons (+48h and +72h).
        </span>
      </div>
    </div>
  );
}
