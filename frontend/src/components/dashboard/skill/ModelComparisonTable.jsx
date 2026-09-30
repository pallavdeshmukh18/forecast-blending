import React from 'react';
import { Table, ShieldCheck, Info } from 'lucide-react';

export default function ModelComparisonTable({ models, unit }) {
  return (
    <div className="overview-card model-comparison-table-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">COMPREHENSIVE BENCHMARK</span>
          <h3 className="overview-card-title">MODEL VERIFICATION METRICS MATRIX</h3>
        </div>
        <div className="table-header-pill">
          <Table size={16} color="#0284c7" />
        </div>
      </div>

      <div className="skill-table-scroll-container">
        <table className="skill-verification-table">
          <thead>
            <tr>
              <th className="th-model">Forecast Model</th>
              <th className="th-num text-right">MAE ({unit})</th>
              <th className="th-num text-right">RMSE ({unit})</th>
              <th className="th-num text-right">Bias ({unit})</th>
              <th className="th-num text-right">Pearson r</th>
              <th className="th-num text-right">CRPS</th>
              <th className="th-num text-right">CRPSS vs Ref</th>
              <th className="th-num text-right">PICP (80%)</th>
              <th className="th-num text-right">Brier Score</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const isBlend = !!m.isBlend;
              return (
                <tr key={m.id} className={`skill-table-row ${isBlend ? 'is-blend-row' : ''}`}>
                  <td className="td-model-info">
                    <div className="model-cell-group">
                      {isBlend && <ShieldCheck size={15} className="blend-icon-table" />}
                      <span className={isBlend ? 'blend-name-highlight' : 'model-name-standard'}>
                        {m.name}
                      </span>
                      <span className="model-type-subtag">{m.type}</span>
                    </div>
                  </td>

                  <td className="td-num text-right monospace">
                    <strong>{m.mae.toFixed(2)}</strong>
                  </td>

                  <td className="td-num text-right monospace">
                    <strong className={isBlend ? 'highlight-blend-rmse' : ''}>
                      {m.rmse.toFixed(2)}
                    </strong>
                  </td>

                  <td className="td-num text-right monospace">
                    <span className={m.bias > 0 ? 'bias-pos' : m.bias < 0 ? 'bias-neg' : ''}>
                      {m.bias > 0 ? `+${m.bias.toFixed(2)}` : m.bias.toFixed(2)}
                    </span>
                  </td>

                  <td className="td-num text-right monospace">
                    {m.corr.toFixed(2)}
                  </td>

                  <td className="td-num text-right monospace">
                    {m.crps.toFixed(2)}
                  </td>

                  <td className="td-num text-right monospace">
                    <span className={m.crpss > 0 ? 'crpss-positive' : m.crpss < 0 ? 'crpss-negative' : ''}>
                      {m.crpss !== undefined && m.crpss !== null
                        ? (m.crpss > 0 ? `+${m.crpss.toFixed(2)}` : m.crpss.toFixed(2))
                        : '—'}
                    </span>
                  </td>

                  <td className="td-num text-right monospace">
                    {m.picp !== undefined ? `${(m.picp * 100).toFixed(1)}%` : '—'}
                  </td>

                  <td className="td-num text-right monospace">
                    {m.brier !== undefined ? m.brier.toFixed(2) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="skill-table-footer-note">
        <Info size={13} className="info-sub-icon" />
        <span>
          Metrics evaluated using standardized WMO verification guidelines. Lower MAE, RMSE, CRPS, and Brier Score indicate superior skill; higher Correlation, CRPSS, and PICP indicate superior skill.
        </span>
      </div>
    </div>
  );
}
