import React from 'react';
import { ArrowUpRight, ArrowDownRight, Layers, Sparkles } from 'lucide-react';

export default function BlendedForecastCard({ forecastData, isLoading }) {
  const locName = forecastData?.location ? `${forecastData.location.name}, ${forecastData.location.state}` : 'Mumbai, Maharashtra';
  const leadHorizon = forecastData?.leadTime ? `${forecastData.leadTime} forecast` : '+24h forecast';
  const blendVal = forecastData?.blendedValue ?? 42;
  const unit = forecastData?.unit ?? 'mm';
  const varLabel = forecastData?.variableLabel ?? 'Rainfall';
  const delta = forecastData?.delta ?? '+8 mm vs previous run';
  const isIncrease = forecastData?.deltaIsIncrease ?? true;

  return (
    <div className="overview-card blended-forecast-hero-card">
      {/* Decorative ambient contour ring like in Neka design */}
      <div className="hero-card-contour-ring"></div>
      <div className="hero-card-glow-orb"></div>

      <div className="hero-card-top">
        <div className="hero-card-header-left">
          <div className="hero-tag-badge">
            <Sparkles size={13} className="hero-sparkle-icon" />
            <span>ADAPTIVE MULTI-MODEL SYNTHESIS</span>
          </div>
          <h2 className="hero-card-title">BLENDED FORECAST</h2>
        </div>

        <div className="hero-card-header-right">
          <div className="hero-location-pill">
            <span className="hero-location-text">{locName}</span>
          </div>
          <div className="hero-lead-pill">
            <span className="hero-lead-text">{leadHorizon}</span>
          </div>
        </div>
      </div>

      <div className="hero-card-body">
        <div className="hero-value-cluster">
          <div className="hero-main-metric">
            <span className="hero-numeric-val">{blendVal}</span>
            <span className="hero-unit-val">{unit}</span>
          </div>
          <div className="hero-variable-name">{varLabel}</div>
        </div>

        <div className="hero-delta-container">
          <div className={`hero-delta-badge ${isIncrease ? 'delta-up' : 'delta-down'}`}>
            {isIncrease ? <ArrowUpRight size={14} strokeWidth={2.5} /> : <ArrowDownRight size={14} strokeWidth={2.5} />}
            <span>{delta}</span>
          </div>
          <span className="hero-delta-context">Calibrated with historical synoptic regime weights</span>
        </div>
      </div>
    </div>
  );
}
