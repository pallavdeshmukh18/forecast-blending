import React from 'react';
import { Clock, Layers, ArrowLeftRight } from 'lucide-react';

export default function UncertaintyByHorizon({ horizonsList, unit, currentLeadTime, onSelectHorizon }) {
  const points = horizonsList || [];
  const activeLead = currentLeadTime ? `+${String(currentLeadTime).replace('+', '').replace('h', '')}h` : '+24h';

  return (
    <div className="overview-card uncertainty-by-horizon-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">LEAD TIME EVOLUTION</span>
          <h3 className="overview-card-title">UNCERTAINTY BY FORECAST HORIZON</h3>
        </div>
        <div className="horizon-table-pill">
          <Clock size={15} color="#0284c7" />
        </div>
      </div>

      <div className="horizon-uncertainty-table-wrap">
        <table className="horizon-uncertainty-table">
          <thead>
            <tr>
              <th className="th-horizon">Horizon</th>
              <th className="th-num text-right">Central Blend ({unit})</th>
              <th className="th-num text-right">Lower Bound (P10)</th>
              <th className="th-num text-right">Upper Bound (P90)</th>
              <th className="th-num text-right">80% Interval Width</th>
              <th className="th-num text-right">Model Spread (σ)</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => {
              const isSelected = p.horizon === activeLead;
              const width = Number((p.high80 - p.low80).toFixed(1));

              return (
                <tr
                  key={p.horizon}
                  className={`horizon-row ${isSelected ? 'is-selected-step' : ''}`}
                  onClick={() => onSelectHorizon && onSelectHorizon(p.horizon.replace('+', '').replace('h', ''))}
                  style={{ cursor: 'pointer' }}
                >
                  <td className="td-horizon">
                    <span className="horizon-step-pill">{p.horizon}</span>
                  </td>

                  <td className="td-num text-right monospace">
                    <strong>{p.forecast.toFixed(1)}</strong>
                  </td>

                  <td className="td-num text-right monospace text-muted">
                    {p.low80.toFixed(1)}
                  </td>

                  <td className="td-num text-right monospace text-muted">
                    {p.high80.toFixed(1)}
                  </td>

                  <td className="td-num text-right monospace">
                    <span className="interval-width-badge">±{(width / 2).toFixed(1)} {unit}</span>
                  </td>

                  <td className="td-num text-right monospace">
                    <span className="spread-badge">{p.spread} {unit}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="horizon-card-footer-note">
        <span>
          Click any horizon row to inspect specific model spreads and quantiles. Error bounds expand monotonically due to chaotic divergence at extended horizons.
        </span>
      </div>
    </div>
  );
}
