import React from 'react';
import { BarChart3, TrendingDown, TrendingUp } from 'lucide-react';

export default function ModelPerformanceCard({ performance, variableLabel }) {
  const models = performance || [
    { id: 'ecmwf', name: 'ECMWF IFS', historical: 8.2, recent: 7.4, isImproved: true },
    { id: 'graphcast', name: 'GraphCast', historical: 9.1, recent: 8.8, isImproved: true },
    { id: 'pangu', name: 'Pangu-Weather', historical: 8.8, recent: 8.1, isImproved: true },
    { id: 'gfs', name: 'NOAA GFS', historical: 11.4, recent: 13.2, isImproved: false },
    { id: 'regional_wrf', name: 'Regional WRF', historical: 10.2, recent: 9.7, isImproved: true },
  ];

  return (
    <div className="overview-card model-performance-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">VERIFICATION ERROR (RMSE)</span>
          <h3 className="overview-card-title">MODEL PERFORMANCE</h3>
        </div>
        <div className="perf-header-icon">
          <BarChart3 size={16} color="#0284c7" />
        </div>
      </div>

      <div className="performance-table-container">
        <table className="performance-table">
          <thead>
            <tr>
              <th className="th-model">Model System</th>
              <th className="th-metric text-right">Historical</th>
              <th className="th-metric text-right">Recent</th>
              <th className="th-trend text-right">Trend</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const diff = (m.recent - m.historical).toFixed(1);
              const isImproved = m.recent < m.historical;
              return (
                <tr key={m.id} className="perf-table-row">
                  <td className="td-model-name">{m.name}</td>
                  <td className="td-metric text-right">{m.historical}</td>
                  <td className="td-metric text-right highlight-metric">{m.recent}</td>
                  <td className="td-trend text-right">
                    <span className={`trend-pill ${isImproved ? 'trend-good' : 'trend-warn'}`}>
                      {isImproved ? (
                        <>
                          <TrendingDown size={12} strokeWidth={2.5} />
                          <span>Improved</span>
                        </>
                      ) : (
                        <>
                          <TrendingUp size={12} strokeWidth={2.5} />
                          <span>Degraded</span>
                        </>
                      )}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="performance-footer-note">
        <span>* Lower RMSE indicates higher predictive accuracy. Historical based on 30-day window; Recent on last 5 cycles.</span>
      </div>
    </div>
  );
}
