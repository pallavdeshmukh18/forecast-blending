import React from 'react';
import { Clock, TrendingUp } from 'lucide-react';

export default function HazardTimeline({ timeline, currentLeadTime, onSelectHorizon, unit }) {
  const points = timeline || [
    { horizon: '+6h', prob: 22, forecast: '12 mm' },
    { horizon: '+12h', prob: 38, forecast: '18 mm' },
    { horizon: '+24h', prob: 72, forecast: '42 mm' },
    { horizon: '+48h', prob: 54, forecast: '31 mm' },
    { horizon: '+72h', prob: 18, forecast: '15 mm' },
  ];

  const activeHorizon = currentLeadTime ? `+${String(currentLeadTime).replace('+', '').replace('h', '')}h` : '+24h';

  return (
    <div className="overview-card hazard-timeline-card-full">
      <div className="overview-card-header">
        <div className="timeline-title-group">
          <Clock size={16} color="#0284c7" />
          <h3 className="overview-card-title">HAZARD TIMELINE</h3>
        </div>
        <span className="timeline-lead-hint">Probability & Magnitude across horizons</span>
      </div>

      <div className="hazard-timeline-table-wrap">
        <table className="hazard-timeline-table">
          <thead>
            <tr>
              <th className="th-horizon">Horizon</th>
              <th className="th-prob-bar">Exceedance Probability</th>
              <th className="th-prob text-right">Probability</th>
              <th className="th-fc text-right">Forecast Magnitude</th>
            </tr>
          </thead>
          <tbody>
            {points.map((pt) => {
              const isSelected = pt.horizon === activeHorizon;
              const probVal = pt.prob || 0;
              const getBarColor = (p) => {
                if (p >= 65) return '#dc2626';
                if (p >= 40) return '#ea580c';
                if (p >= 20) return '#eab308';
                return '#10b981';
              };
              const color = getBarColor(probVal);

              return (
                <tr
                  key={pt.horizon}
                  className={`hazard-timeline-row ${isSelected ? 'is-active-step' : ''}`}
                  onClick={() => onSelectHorizon && onSelectHorizon(pt.horizon.replace('+', '').replace('h', ''))}
                >
                  <td className="td-horizon">
                    <span className="horizon-badge-pill">{pt.horizon}</span>
                  </td>

                  <td className="td-prob-bar">
                    <div className="timeline-prog-track">
                      <div
                        className="timeline-prog-fill"
                        style={{
                          width: `${probVal}%`,
                          backgroundColor: color,
                        }}
                      ></div>
                    </div>
                  </td>

                  <td className="td-prob text-right">
                    <span className="timeline-prob-val" style={{ color }}>
                      {probVal}%
                    </span>
                  </td>

                  <td className="td-fc text-right monospace">
                    <strong>{pt.forecast}</strong>
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
