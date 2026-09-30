import React from 'react';
import { Activity, ShieldCheck } from 'lucide-react';

export default function ModelForecastsMiniCard({ forecastsData, unit }) {
  const models = forecastsData?.models || [
    { id: 'ecmwf', name: 'ECMWF IFS', value: 45 },
    { id: 'graphcast', name: 'GraphCast', value: 39 },
    { id: 'pangu', name: 'Pangu-Weather', value: 43 },
    { id: 'gfs', name: 'NOAA GFS', value: 51 },
    { id: 'regional_wrf', name: 'Regional WRF', value: 37 },
  ];

  const blendValue = forecastsData?.blendValue ?? 42;
  const unitLabel = unit || 'mm';

  return (
    <div className="overview-card model-forecasts-mini-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">PREDICTION SPREAD</span>
          <h3 className="overview-card-title">MODEL FORECASTS</h3>
        </div>
        <div className="forecasts-mini-icon">
          <Activity size={16} color="#0284c7" />
        </div>
      </div>

      <div className="forecasts-mini-list">
        {models.map((m) => (
          <div key={m.id} className="forecast-mini-row">
            <span className="forecast-mini-name">{m.name}</span>
            <span className="forecast-mini-val">
              <strong>{m.value}</strong> {unitLabel}
            </span>
          </div>
        ))}
      </div>

      <div className="forecasts-mini-divider"></div>

      <div className="forecasts-mini-blend-row">
        <div className="blend-mini-title">
          <ShieldCheck size={16} className="blend-shield-icon" />
          <span className="blend-brand-text">SkyCast Blend</span>
        </div>
        <span className="blend-mini-val">
          <strong>{blendValue}</strong> {unitLabel}
        </span>
      </div>
    </div>
  );
}
