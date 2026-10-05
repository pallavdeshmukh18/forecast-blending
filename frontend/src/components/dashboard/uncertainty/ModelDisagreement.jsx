import React from 'react';
import { Layers, ShieldCheck, ArrowLeftRight } from 'lucide-react';

export default function ModelDisagreement({ models, unit, modelSpread }) {
  const modelList = models || [];
  const blend = modelList.find((m) => m.isBlend) || modelList[0];

  // Calculate min and max forecast across models for relative alignment
  const vals = modelList.map((m) => m.forecast);
  const minVal = Math.min(...vals, 0);
  const maxVal = Math.max(...vals, 1);
  const valRange = maxVal - minVal || 1;

  return (
    <div className="overview-card model-disagreement-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">MULTI-SYSTEM DISPERSION</span>
          <h3 className="overview-card-title">MODEL FORECAST SPREAD & DISAGREEMENT</h3>
        </div>
        <div className="disagreement-header-pill">
          <Layers size={15} color="#0284c7" />
          <span>Spread: {modelSpread} {unit}</span>
        </div>
      </div>

      <div className="disagreement-table-container">
        <table className="disagreement-table">
          <thead>
            <tr>
              <th className="th-system">Model System</th>
              <th className="th-val text-right">Forecast ({unit})</th>
              <th className="th-dispersion">Dispersion vs Blend</th>
              <th className="th-delta text-right">Departure (Δ)</th>
            </tr>
          </thead>
          <tbody>
            {modelList.map((m) => {
              const isBlend = !!m.isBlend;
              const delta = m.delta ?? (m.forecast - (blend?.forecast || 0));
              const posPct = Math.max(0, Math.min(100, ((m.forecast - minVal) / valRange) * 100));

              return (
                <tr key={m.id} className={`disagreement-row ${isBlend ? 'is-blend-row' : ''}`}>
                  <td className="td-system">
                    <div className="system-cell-group">
                      {isBlend && <ShieldCheck size={14} className="blend-icon" />}
                      <span className={isBlend ? 'blend-name-text' : 'model-name-text'}>
                        {m.name}
                      </span>
                    </div>
                  </td>

                  <td className="td-val text-right monospace">
                    <strong className={isBlend ? 'highlight-blend-val' : ''}>
                      {m.forecast.toFixed(1)}
                    </strong>
                  </td>

                  <td className="td-dispersion">
                    <div className="dispersion-track">
                      {/* Center zero line marker representing blend */}
                      <div
                        className="dispersion-center-marker"
                        style={{
                          left: `${Math.max(0, Math.min(100, (((blend?.forecast || 0) - minVal) / valRange) * 100))}%`,
                        }}
                      ></div>

                      {/* Model forecast dot */}
                      <div
                        className={`dispersion-dot ${isBlend ? 'blend-dot' : ''}`}
                        style={{ left: `${posPct}%` }}
                        title={`${m.name}: ${m.forecast} ${unit}`}
                      ></div>
                    </div>
                  </td>

                  <td className="td-delta text-right monospace">
                    {isBlend ? (
                      <span className="delta-zero">Baseline [0.0]</span>
                    ) : (
                      <span className={delta > 0 ? 'delta-pos' : delta < 0 ? 'delta-neg' : 'delta-zero'}>
                        {delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)} {unit}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="disagreement-footer-explainer">
        <span>
          Disagreement reflects cross-model standard deviation. A compact cluster around the BharatWeather blend indicates robust physical consensus, while wider dispersion flags sensitivity to atmospheric initial conditions.
        </span>
      </div>
    </div>
  );
}
