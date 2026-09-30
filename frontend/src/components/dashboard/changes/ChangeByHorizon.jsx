import React from 'react';
import { Clock, TrendingUp, ArrowUpRight, ArrowDownRight, ArrowRight } from 'lucide-react';

export default function ChangeByHorizon({ horizonChanges, unit, currentLeadTime, onSelectHorizon }) {
  const horizons = horizonChanges || [];
  const activeLead = currentLeadTime ? `+${String(currentLeadTime).replace('+', '').replace('h', '')}h` : '+24h';

  return (
    <div className="overview-card change-by-horizon-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">LEAD TIME PROGRESSION</span>
          <h3 className="overview-card-title">FORECAST EVOLUTION ACROSS HORIZONS</h3>
        </div>
        <div className="horizon-pill">
          <Clock size={15} color="#0284c7" />
        </div>
      </div>

      <div className="horizon-changes-table-wrap">
        <table className="horizon-changes-table">
          <thead>
            <tr>
              <th className="th-horizon">Horizon</th>
              <th className="th-num text-right">Previous Run ({unit})</th>
              <th className="th-num text-right">Current Run ({unit})</th>
              <th className="th-delta text-right">Net Change (Δ)</th>
              <th className="th-spread text-right">Spread Evolution</th>
            </tr>
          </thead>
          <tbody>
            {horizons.map((h) => {
              const isSelected = h.horizon === activeLead;
              const isUp = h.change > 0;
              const isDown = h.change < 0;
              const DirectionIcon = isUp ? ArrowUpRight : isDown ? ArrowDownRight : ArrowRight;

              return (
                <tr
                  key={h.horizon}
                  className={`horizon-change-row ${isSelected ? 'is-selected-step' : ''}`}
                  onClick={() => onSelectHorizon && onSelectHorizon(h.horizon.replace('+', '').replace('h', ''))}
                  style={{ cursor: 'pointer' }}
                >
                  <td className="td-horizon">
                    <span className="horizon-step-pill">{h.horizon}</span>
                  </td>

                  <td className="td-num text-right monospace text-muted">
                    {h.previous.toFixed(1)}
                  </td>

                  <td className="td-num text-right monospace">
                    <strong className="current-highlight">{h.current.toFixed(1)}</strong>
                  </td>

                  <td className="td-delta text-right">
                    <span className={`delta-tag ${isUp ? 'delta-up' : isDown ? 'delta-down' : 'delta-zero'}`}>
                      <DirectionIcon size={12} strokeWidth={2.5} />
                      <span>{h.change > 0 ? `+${h.change.toFixed(1)}` : h.change.toFixed(1)} {unit}</span>
                    </span>
                  </td>

                  <td className="td-spread text-right monospace text-muted">
                    <span>{h.prevSpread} → <strong>{h.currSpread}</strong> {unit}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="horizon-card-footer-note">
        <span>
          Click any lead time step to adjust the active analysis horizon. Inter-run shifts tend to widen at +48h and +72h due to progressive phase discrepancies in wave propagation.
        </span>
      </div>
    </div>
  );
}
