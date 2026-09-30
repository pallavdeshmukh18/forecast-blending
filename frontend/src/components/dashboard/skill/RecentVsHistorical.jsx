import React from 'react';
import { History, TrendingDown, TrendingUp, ShieldCheck } from 'lucide-react';

export default function RecentVsHistorical({ models, unit }) {
  return (
    <div className="overview-card recent-vs-historical-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">TEMPORAL STABILITY</span>
          <h3 className="overview-card-title">RECENT VS HISTORICAL PERFORMANCE</h3>
        </div>
        <div className="recent-icon-pill">
          <History size={16} color="#0284c7" />
        </div>
      </div>

      <div className="recent-table-wrap">
        <table className="recent-hist-table">
          <thead>
            <tr>
              <th className="th-model">Model</th>
              <th className="th-rmse text-right">Historical RMSE ({unit})</th>
              <th className="th-rmse text-right">Recent RMSE ({unit})</th>
              <th className="th-change text-right">Change</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const isBlend = !!m.isBlend;
              const chg = m.rmseChangePct || 0;
              const isImprovement = chg < 0; // Negative change in RMSE is an improvement

              return (
                <tr key={m.id} className={`recent-row ${isBlend ? 'is-blend-row' : ''}`}>
                  <td className="td-model">
                    <div className="recent-model-name-group">
                      {isBlend && <ShieldCheck size={14} className="blend-icon-table" />}
                      <span className={isBlend ? 'blend-name-highlight' : 'model-name-standard'}>
                        {m.name}
                      </span>
                    </div>
                  </td>

                  <td className="td-rmse text-right monospace">
                    {m.historicalRmse ? m.historicalRmse.toFixed(2) : '—'}
                  </td>

                  <td className="td-rmse text-right monospace">
                    <strong>{m.recentRmse ? m.recentRmse.toFixed(2) : m.rmse.toFixed(2)}</strong>
                  </td>

                  <td className="td-change text-right">
                    <span
                      className={`trend-pill ${
                        isImprovement ? 'trend-good' : chg === 0 ? 'trend-neutral' : 'trend-warn'
                      }`}
                    >
                      {isImprovement ? (
                        <TrendingDown size={11} strokeWidth={2.5} />
                      ) : (
                        <TrendingUp size={11} strokeWidth={2.5} />
                      )}
                      <span>
                        {chg > 0 ? `+${chg.toFixed(1)}%` : `${chg.toFixed(1)}%`}
                      </span>
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="recent-card-footer-note">
        <span>
          Recent evaluation reflects a 30-day walk-forward window vs historical seasonal baseline. Negative change values denote reduced error (skill improvement).
        </span>
      </div>
    </div>
  );
}
