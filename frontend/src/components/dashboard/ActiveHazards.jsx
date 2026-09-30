import React from 'react';
import { AlertTriangle, ArrowRight, CloudRain, Wind, Flame } from 'lucide-react';

export default function ActiveHazards({ hazardsData, onNavigateToExtremes }) {
  const hazards = hazardsData?.hazards || [
    {
      id: 'rain',
      name: 'Heavy Rain',
      regions: '3 regions affected',
      severity: 'Warning',
      icon: CloudRain,
      color: '#dc2626',
      bg: '#fef2f2',
      border: '#fecaca',
    },
    {
      id: 'wind',
      name: 'High Wind',
      regions: '2 regions affected',
      severity: 'Watch',
      icon: Wind,
      color: '#d97706',
      bg: '#fffbeb',
      border: '#fde68a',
    },
    {
      id: 'heat',
      name: 'Heat',
      regions: '1 region affected',
      severity: 'Advisory',
      icon: Flame,
      color: '#ea580c',
      bg: '#fff7ed',
      border: '#ffedd5',
    },
  ];

  const getHazardIcon = (hazard) => {
    if (hazard?.icon) return hazard.icon;
    const key = (hazard?.id || hazard?.name || '').toLowerCase();
    if (key.includes('rain')) return CloudRain;
    if (key.includes('wind')) return Wind;
    if (key.includes('heat') || key.includes('temp')) return Flame;
    return AlertTriangle;
  };

  return (
    <div className="overview-card active-hazards-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">WARNINGS</span>
          <h3 className="overview-card-title">ACTIVE HAZARDS</h3>
        </div>

        <button
          type="button"
          className="hazards-view-all-btn"
          onClick={onNavigateToExtremes}
          title="Explore Extreme Weather Details"
        >
          <span>View all</span>
          <ArrowRight size={14} />
        </button>
      </div>

      <div className="hazards-summary-list">
        {hazards.map((h) => {
          const Icon = getHazardIcon(h);
          return (
            <div
              key={h.id || h.name}
              className="hazard-summary-item"
              style={{ background: h.bg || '#fef2f2', borderColor: h.border || '#fecaca' }}
            >
              <div className="hazard-item-left">
                <div className="hazard-warning-icon" style={{ color: h.color || '#dc2626' }}>
                  <AlertTriangle size={16} strokeWidth={2.2} />
                </div>
                <div className="hazard-text-block">
                  <div className="hazard-item-title-row">
                    <span className="hazard-name">{h.name}</span>
                    <span className="hazard-severity-tag" style={{ color: h.color || '#dc2626' }}>
                      {h.severity}
                    </span>
                  </div>
                  <span className="hazard-regions-count">{h.regions || `${h.regions_affected || 1} regions affected`}</span>
                </div>
              </div>

              <div className="hazard-mini-icon" style={{ color: h.color || '#dc2626' }}>
                {Icon && <Icon size={16} strokeWidth={2} />}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
