import React from 'react';
import { ChevronDown, ArrowUp, ArrowDown, Droplets, Thermometer, Wind } from 'lucide-react';

export default function ForecastSummary({ leadTime, onLeadTimeChange, data }) {
  const metrics = [
    {
      id: 'rainfall',
      name: 'Rainfall',
      value: data?.rainfall?.value ?? '42 mm',
      change: data?.rainfall?.change ?? '↑ 8 mm',
      isIncrease: true,
      icon: Droplets,
      iconColor: '#0284c7',
      iconBg: '#e0f2fe',
    },
    {
      id: 'temperature',
      name: 'Temperature',
      value: data?.temperature?.value ?? '31.4°C',
      change: data?.temperature?.change ?? '↓ 0.6°C',
      isIncrease: false,
      icon: Thermometer,
      iconColor: '#ea580c',
      iconBg: '#ffedd5',
    },
    {
      id: 'wind',
      name: 'Wind',
      value: data?.wind?.value ?? '28 km/h',
      change: data?.wind?.change ?? '↑ 4 km/h',
      isIncrease: true,
      icon: Wind,
      iconColor: '#059669',
      iconBg: '#dcfce7',
    },
  ];

  return (
    <div className="overview-card forecast-summary-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">PREDICTION</span>
          <h3 className="overview-card-title">CURRENT BLENDED FORECAST</h3>
        </div>

        {/* Lead Time Dropdown Selector */}
        <div className="lead-time-picker">
          <select
            value={leadTime}
            onChange={(e) => onLeadTimeChange(e.target.value)}
            className="lead-time-select"
            aria-label="Select forecast horizon"
          >
            <option value="6h">+6h</option>
            <option value="12h">+12h</option>
            <option value="24h">+24h</option>
            <option value="48h">+48h</option>
            <option value="72h">+72h</option>
          </select>
          <ChevronDown size={14} className="lead-time-chevron" />
        </div>
      </div>

      <div className="forecast-metrics-list">
        {metrics.map((m) => {
          const Icon = m.icon;
          return (
            <div key={m.id} className="forecast-metric-row">
              <div className="metric-left">
                <div className="metric-icon-box" style={{ background: m.iconBg, color: m.iconColor }}>
                  <Icon size={16} strokeWidth={2.2} />
                </div>
                <span className="metric-name">{m.name}</span>
              </div>

              <div className="metric-values">
                <span className="metric-main-val">{m.value}</span>
                {(() => {
                  const isUp = m.change.includes('↑') || m.change.includes('+');
                  const isDown = m.change.includes('↓') || m.change.includes('-');
                  const cleanText = m.change.replace(/^[↑↓]\s*/, '');
                  return (
                    <span className={`metric-change-badge ${isUp ? 'up' : isDown ? 'down' : 'neutral'}`}>
                      {isUp && <ArrowUp size={12} strokeWidth={2.5} />}
                      {isDown && <ArrowDown size={12} strokeWidth={2.5} />}
                      <span>{cleanText}</span>
                    </span>
                  );
                })()}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
