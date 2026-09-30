import React from 'react';
import { CloudRain, Thermometer, Wind, ShieldCheck, ArrowUpRight, ArrowDownRight } from 'lucide-react';

export default function OverviewKpiCards({ data }) {
  const cards = [
    {
      id: 'rainfall',
      label: 'RAINFALL RISK',
      primaryValue: data?.rainfallRisk?.percent ?? '68%',
      secondaryValue: data?.rainfallRisk?.level ?? 'High',
      icon: CloudRain,
      accentColor: '#0284c7',
      bgGradient: 'linear-gradient(135deg, rgba(2, 132, 199, 0.08) 0%, rgba(56, 189, 248, 0.03) 100%)',
      borderColor: 'rgba(56, 189, 248, 0.3)',
      badgeColor: '#e0f2fe',
      badgeTextColor: '#0369a1',
    },
    {
      id: 'temperature',
      label: 'TEMPERATURE',
      primaryValue: data?.temperature?.value ?? '34.8°C',
      secondaryValue: data?.temperature?.change ?? '+1.2°C',
      isPositiveChange: true,
      icon: Thermometer,
      accentColor: '#ea580c',
      bgGradient: 'linear-gradient(135deg, rgba(234, 88, 12, 0.08) 0%, rgba(251, 146, 60, 0.03) 100%)',
      borderColor: 'rgba(251, 146, 60, 0.3)',
      badgeColor: '#ffedd5',
      badgeTextColor: '#c2410c',
    },
    {
      id: 'wind',
      label: 'WIND',
      primaryValue: data?.wind?.value ?? '42 km/h',
      secondaryValue: data?.wind?.level ?? 'Moderate',
      icon: Wind,
      accentColor: '#10b981',
      bgGradient: 'linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(52, 211, 153, 0.03) 100%)',
      borderColor: 'rgba(52, 211, 153, 0.3)',
      badgeColor: '#ecfdf5',
      badgeTextColor: '#047857',
    },
    {
      id: 'confidence',
      label: 'FORECAST CONFIDENCE',
      primaryValue: data?.confidence?.percent ?? '87%',
      secondaryValue: data?.confidence?.status ?? 'High agreement',
      icon: ShieldCheck,
      accentColor: '#6366f1',
      bgGradient: 'linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(129, 140, 248, 0.03) 100%)',
      borderColor: 'rgba(129, 140, 248, 0.3)',
      badgeColor: '#eef2ff',
      badgeTextColor: '#4338ca',
    },
  ];

  return (
    <div className="overview-kpi-grid">
      {cards.map((card) => {
        const IconComponent = card.icon;
        return (
          <div
            key={card.id}
            className="overview-kpi-card"
            style={{
              background: card.bgGradient,
              borderColor: card.borderColor,
            }}
          >
            {/* Subtle isobar/contour decorative geometric ring like in Neka design */}
            <div className="kpi-contour-ring" style={{ borderColor: card.accentColor }}></div>

            <div className="kpi-top-row">
              <span className="kpi-label">{card.label}</span>
              <div className="kpi-icon-pill" style={{ color: card.accentColor, background: card.badgeColor }}>
                <IconComponent size={15} strokeWidth={2.2} />
              </div>
            </div>

            <div className="kpi-body">
              <div className="kpi-primary-val">{card.primaryValue}</div>
              <div
                className="kpi-secondary-badge"
                style={{
                  background: card.badgeColor,
                  color: card.badgeTextColor,
                }}
              >
                {card.isPositiveChange && <ArrowUpRight size={13} strokeWidth={2.5} />}
                <span>{card.secondaryValue}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
