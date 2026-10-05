import React from 'react';
import { Layers, ShieldCheck } from 'lucide-react';

export default function ExtremeModelAgreement({ models, unit }) {
  const modelList = models || [
    { id: 'ecmwf', name: 'ECMWF IFS', prob: 68, forecast: '45 mm' },
    { id: 'gfs', name: 'NOAA GFS', prob: 82, forecast: '51 mm' },
    { id: 'graphcast', name: 'GraphCast', prob: 55, forecast: '39 mm' },
    { id: 'pangu', name: 'Pangu-Weather', prob: 62, forecast: '43 mm' },
    { id: 'regional_wrf', name: 'Regional WRF', prob: 50, forecast: '37 mm' },
    { id: 'skycast', name: 'BharatWeather Blend', prob: 72, forecast: '42 mm' },
  ];

  return (
    <div className="overview-card extreme-model-agreement-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">MULTI-SYSTEM SPREAD</span>
          <h3 className="overview-card-title">MODEL AGREEMENT</h3>
        </div>
        <div className="agreement-icon-pill">
          <Layers size={16} color="#0284c7" />
        </div>
      </div>

      <div className="extreme-table-container">
        <table className="extreme-models-table">
          <thead>
            <tr>
              <th className="th-system">Model System</th>
              <th className="th-prob text-right">Event Probability</th>
              <th className="th-forecast text-right">Forecast Value</th>
            </tr>
          </thead>
          <tbody>
            {modelList.map((m) => {
              const isBlend = m.id === 'skycast';
              return (
                <tr key={m.id} className={`extreme-model-row ${isBlend ? 'highlight-blend-row' : ''}`}>
                  <td className="td-system">
                    <div className="system-name-cell">
                      {isBlend && <ShieldCheck size={14} className="blend-td-icon" />}
                      <span className={isBlend ? 'blend-name-text' : 'model-name-text'}>{m.name}</span>
                    </div>
                  </td>

                  <td className="td-prob text-right">
                    <div className="prob-bar-cell">
                      <div className="prob-micro-track">
                        <div
                          className={`prob-micro-fill ${isBlend ? 'blend-fill' : ''}`}
                          style={{ width: `${m.prob}%` }}
                        ></div>
                      </div>
                      <span className="prob-number-tag">{m.prob}%</span>
                    </div>
                  </td>

                  <td className="td-forecast text-right monospace">
                    <strong>{m.forecast}</strong>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
