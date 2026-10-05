import React from 'react';
import { Target, BarChart2, AlertTriangle, ShieldCheck } from 'lucide-react';

const WHY_CARDS = [
  {
    title: 'Blended Forecasts',
    desc: 'Best estimate from multiple global models.',
    icon: <Target size={26} color="#2563eb" strokeWidth={2.2} />,
    bgColor: '#eff6ff'
  },
  {
    title: 'Model Insights',
    desc: 'See which models are more reliable and why.',
    icon: <BarChart2 size={26} color="#7c3aed" strokeWidth={2.2} />,
    bgColor: '#f5f3ff'
  },
  {
    title: 'Extreme Weather Alerts',
    desc: 'Early signals for heavy rain, heat waves and high winds.',
    icon: <AlertTriangle size={26} color="#dc2626" strokeWidth={2.2} />,
    bgColor: '#fef2f2'
  },
  {
    title: 'Built for India',
    desc: 'Location-specific forecasts for a safer and more resilient future.',
    icon: <ShieldCheck size={26} color="#059669" strokeWidth={2.2} />,
    bgColor: '#ecfdf5'
  }
];

export default function WhySkyCastSection() {
  return (
    <section className="why-section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">WHY BHARATWEATHER</span>
          <h2 className="section-title">Why BharatWeather: More Than Just a Forecast</h2>
          <p className="section-subtitle">Deeper insights for better decisions.</p>
        </div>

        <div className="why-cards-grid">
          {WHY_CARDS.map((card, idx) => (
            <div key={idx} className="why-card">
              <div className="why-card-icon-box" style={{ background: card.bgColor }}>
                {card.icon}
              </div>
              <h3 className="why-card-title">{card.title}</h3>
              <p className="why-card-desc">{card.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
