import React from 'react';
import { CloudRain, Flame, Wind, ArrowRight, ShieldCheck, AlertCircle } from 'lucide-react';

export default function HazardOverviewCards({ cards, selectedHazardId, onSelectHazard }) {
  const getIcon = (id) => {
    switch (id) {
      case 'heavy_rain':
        return CloudRain;
      case 'heatwave':
        return Flame;
      case 'high_wind':
        return Wind;
      default:
        return AlertCircle;
    }
  };

  return (
    <div className="hazard-cards-grid">
      {cards.map((card) => {
        const IconComponent = getIcon(card.id);
        const isSelected = selectedHazardId === card.id;

        return (
          <div
            key={card.id}
            className={`hazard-overview-card ${isSelected ? 'is-card-selected' : ''}`}
            onClick={() => onSelectHazard(card.id)}
            style={{
              borderColor: isSelected ? card.color : undefined,
            }}
          >
            {/* Top Row: Icon + Name + Severity Badge */}
            <div className="hoc-top-row">
              <div className="hoc-left-group">
                <div
                  className="hoc-icon-pill"
                  style={{ color: card.color, background: card.bg }}
                >
                  <IconComponent size={18} strokeWidth={2.2} />
                </div>
                <div>
                  <h4 className="hoc-name">{card.name}</h4>
                  <span className="hoc-threshold-tag">{card.threshold}</span>
                </div>
              </div>

              <div
                className="hoc-severity-badge"
                style={{
                  color: card.color,
                  background: card.bg,
                  borderColor: card.border,
                }}
              >
                <span>{card.severity}</span>
              </div>
            </div>

            {/* Middle: Calibrated Probability & Forecast Value */}
            <div className="hoc-metrics-body">
              <div className="hoc-metric-col">
                <span className="hoc-stat-label">CALIBRATED PROB</span>
                <span className="hoc-prob-value" style={{ color: card.color }}>
                  {card.probability}
                </span>
              </div>

              <div className="hoc-metric-col text-right">
                <span className="hoc-stat-label">FORECAST VALUE</span>
                <span className="hoc-forecast-val">{card.forecastValue}</span>
              </div>
            </div>

            {/* Bottom: Horizon + Confidence + Selection Indicator */}
            <div className="hoc-bottom-row">
              <div className="hoc-conf-group">
                <ShieldCheck size={13} className="hoc-shield-icon" />
                <span>Confidence: <strong>{card.confidence}</strong></span>
              </div>

              <div className="hoc-action-link" style={{ color: isSelected ? card.color : undefined }}>
                <span>{isSelected ? 'Active Selection' : 'Inspect'}</span>
                <ArrowRight size={13} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
