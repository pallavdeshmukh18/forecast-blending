import React, { useState } from 'react';
import { Cloud, Layers, Cpu, Compass, Globe2, Activity } from 'lucide-react';

const MODELS = [
  {
    id: 'ecmwf',
    name: 'ECMWF',
    category: 'Global NWP',
    weight: '30% Weight',
    iconBg: '#e0f2fe',
    iconColor: '#0369a1',
    description: 'European Centre for Medium-Range Weather Forecasts (IFS high-resolution physics model)',
    svgIcon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0284c7" strokeWidth="2.5">
        <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V9s-1 1-4 1-5-2-8-2-4 1-4 1v6z" />
        <path d="M4 9a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" />
      </svg>
    )
  },
  {
    id: 'graphcast',
    name: 'GraphCast',
    category: 'AI Model',
    weight: '28% Weight',
    iconBg: '#fef3c7',
    iconColor: '#d97706',
    description: 'DeepMind Graph Neural Network delivering 10-day forecasts at unprecedented resolution',
    svgIcon: (
      <svg width="24" height="24" viewBox="0 0 24 24">
        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" fill="#ea4335" />
        <path d="M12 6a6 6 0 0 1 4.24 1.76l-1.42 1.42A4 4 0 1 0 12 16v-2a2 2 0 1 1 0-4h4v2h-4z" fill="#4285f4" />
      </svg>
    )
  },
  {
    id: 'pangu',
    name: 'Pangu-Weather',
    category: 'AI Model',
    weight: '14% Weight',
    iconBg: '#fee2e2',
    iconColor: '#dc2626',
    description: '3D Earth-Specific Transformer model with strong convective & cyclone tracking capabilities',
    svgIcon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="#dc2626">
        <circle cx="12" cy="7" r="3" />
        <circle cx="7" cy="14" r="3" />
        <circle cx="17" cy="14" r="3" />
        <circle cx="12" cy="19" r="2.5" />
      </svg>
    )
  },
  {
    id: 'gfs',
    name: 'GFS',
    category: 'Global NWP',
    weight: '18% Weight',
    iconBg: '#dbeafe',
    iconColor: '#1d4ed8',
    description: 'NOAA Global Forecast System numerical weather prediction with 13km horizontal grid',
    svgIcon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1d4ed8" strokeWidth="2.2">
        <circle cx="12" cy="12" r="10" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
      </svg>
    )
  },
  {
    id: 'regional',
    name: 'Regional Models',
    category: 'High Resolution',
    weight: '10% Weight',
    iconBg: '#ccfbf1',
    iconColor: '#0d9488',
    description: 'IMD WRF & NCMRWF regional high-resolution convective resolving meshes for Indian orography',
    svgIcon: (
      <Layers size={22} color="#0d9488" strokeWidth={2.2} />
    )
  }
];

export default function ApproachSection() {
  const [selectedModel, setSelectedModel] = useState(null);

  return (
    <section id="about" className="approach-section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">OUR APPROACH</span>
          <h2 className="section-title">
            BharatWeather Blending Architecture: the Best of Global Weather Models
          </h2>
          <p className="section-subtitle">
            We intelligently blend multiple AI and numerical weather models,
            adapting to region, season and weather conditions to give you the most
            reliable forecast.
          </p>
        </div>

        <div className="approach-diagram">
          {/* Models Top Row */}
          <div className="models-row">
            {MODELS.map((model) => (
              <div
                key={model.id}
                className={`model-card ${selectedModel?.id === model.id ? 'active' : ''}`}
                onClick={() => setSelectedModel(model)}
                onMouseEnter={() => setSelectedModel(model)}
              >
                <div className="model-logo-box" style={{ background: model.iconBg }}>
                  {model.svgIcon}
                </div>
                <h4 className="model-name">{model.name}</h4>
                <span className="model-category">{model.category}</span>
                <span className="model-weight-pill">{model.weight}</span>
              </div>
            ))}
          </div>

          {/* SVG Animated Connecting Lines */}
          <svg className="approach-svg-overlay" viewBox="0 0 1000 120" preserveAspectRatio="none">
            {/* Curves from the 5 columns to center (500, 100) */}
            {/* Model 1: ~100 to 500 */}
            <path d="M 100 0 C 100 65, 460 70, 485 95" className="flow-path" />
            {/* Model 2: ~300 to 500 */}
            <path d="M 300 0 C 300 55, 470 65, 492 95" className="flow-path" />
            {/* Model 3: ~500 to 500 */}
            <path d="M 500 0 C 500 40, 500 60, 500 95" className="flow-path" />
            {/* Model 4: ~700 to 500 */}
            <path d="M 700 0 C 700 55, 530 65, 508 95" className="flow-path" />
            {/* Model 5: ~900 to 500 */}
            <path d="M 900 0 C 900 65, 540 70, 515 95" className="flow-path" />
          </svg>

          {/* Center Hub */}
          <div className="center-node-container">
            <div className="center-hub-circle">
              <Cloud size={38} fill="white" strokeWidth={1.5} />
            </div>
            <div className="center-hub-label">
              <h3 className="center-hub-title">BharatWeather</h3>
              <p className="center-hub-subtitle">Intelligent Blending</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
