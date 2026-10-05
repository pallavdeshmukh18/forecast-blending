import React from 'react';
import { Cpu, ShieldCheck } from 'lucide-react';

export default function ModelForecastComparison({ forecastData }) {
  const models = forecastData?.models || [
    { id: 'ecmwf', name: 'ECMWF IFS', value: 45, unit: 'mm', color: '#0284c7' },
    { id: 'graphcast', name: 'GraphCast', value: 39, unit: 'mm', color: '#38bdf8' },
    { id: 'pangu', name: 'Pangu-Weather', value: 43, unit: 'mm', color: '#818cf8' },
    { id: 'gfs', name: 'NOAA GFS', value: 51, unit: 'mm', color: '#f59e0b' },
    { id: 'regional_wrf', name: 'Regional WRF', value: 37, unit: 'mm', color: '#10b981' },
  ];

  const blendVal = forecastData?.blendedValue ?? 42;
  const unit = forecastData?.unit ?? 'mm';

  // Calculate scaling for horizontal bars relative to max value
  const allValues = [...models.map((m) => m.value), blendVal];
  const maxVal = Math.max(...allValues, 1);

  return (
    <div className="overview-card model-forecasts-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">MULTI-SYSTEM INGESTION</span>
          <h3 className="overview-card-title">MODEL FORECASTS</h3>
        </div>
        <div className="models-count-badge">
          <Cpu size={14} />
          <span>5 Models Evaluated</span>
        </div>
      </div>

      <div className="model-comparison-bars-container">
        {/* Individual member models */}
        <div className="models-bars-list">
          {models.map((m) => {
            const barWidthPercent = Math.min(100, Math.max(8, (m.value / (maxVal * 1.08)) * 100));
            return (
              <div key={m.id} className="model-bar-row">
                <div className="model-bar-header">
                  <span className="model-name-label">{m.name}</span>
                  <span className="model-val-label">
                    <strong>{m.value}</strong> {unit}
                  </span>
                </div>
                <div className="model-bar-track">
                  <div
                    className="model-bar-fill"
                    style={{
                      width: `${barWidthPercent}%`,
                      background: m.color || '#3b82f6',
                    }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Distinct operational divider */}
        <div className="model-comparison-divider">
          <span>OPERATIONAL BLEND RESULT</span>
        </div>

        {/* BharatWeather Blended Composite */}
        <div className="blend-bar-row">
          <div className="model-bar-header">
            <div className="blend-title-group">
              <ShieldCheck size={16} className="blend-check-icon" />
              <span className="blend-name-label">BHARATWEATHER BLEND</span>
              <span className="blend-tag-pill">Convex Optimal</span>
            </div>
            <span className="blend-val-label">
              <strong>{blendVal}</strong> {unit}
            </span>
          </div>
          <div className="model-bar-track blend-bar-track">
            <div
              className="model-bar-fill blend-bar-fill"
              style={{
                width: `${Math.min(100, Math.max(8, (blendVal / (maxVal * 1.08)) * 100))}%`,
              }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
}
