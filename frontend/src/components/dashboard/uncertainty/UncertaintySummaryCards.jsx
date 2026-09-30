import React from 'react';
import { Target, Layers, ArrowLeftRight, ShieldCheck, Binary } from 'lucide-react';

export default function UncertaintySummaryCards({ data }) {
  const central = data?.centralForecast ?? 42.0;
  const unit = data?.unit || 'mm';
  const [low80, high80] = data?.predictionInterval80 || [32.0, 54.0];
  const spread = data?.modelSpread ?? 5.1;
  const confidence = data?.confidenceRating || 'High Confidence';
  const intervalWidth = data?.intervalWidth80 ?? (high80 - low80);

  return (
    <div className="uncertainty-summary-grid">
      {/* 1. Central Forecast */}
      <div className="uncertainty-kpi-card">
        <div className="ukpi-header">
          <span className="ukpi-label">Central Forecast</span>
          <div className="ukpi-icon-pill icon-blue">
            <Target size={15} />
          </div>
        </div>
        <div className="ukpi-val-row">
          <span className="ukpi-main-number">{central}</span>
          <span className="ukpi-unit">{unit}</span>
        </div>
        <span className="ukpi-sub">Blended expected value</span>
      </div>

      {/* 2. Prediction Range (80%) */}
      <div className="uncertainty-kpi-card">
        <div className="ukpi-header">
          <span className="ukpi-label">80% Prediction Range</span>
          <div className="ukpi-icon-pill icon-emerald">
            <ArrowLeftRight size={15} />
          </div>
        </div>
        <div className="ukpi-val-row">
          <span className="ukpi-range-text">
            {low80} — {high80}
          </span>
          <span className="ukpi-unit">{unit}</span>
        </div>
        <span className="ukpi-sub">Width: ±{(intervalWidth / 2).toFixed(1)} {unit}</span>
      </div>

      {/* 3. Model Spread (Std Dev) */}
      <div className="uncertainty-kpi-card">
        <div className="ukpi-header">
          <span className="ukpi-label">Model Disagreement Spread</span>
          <div className="ukpi-icon-pill icon-amber">
            <Layers size={15} />
          </div>
        </div>
        <div className="ukpi-val-row">
          <span className="ukpi-main-number">{spread}</span>
          <span className="ukpi-unit">{unit}</span>
        </div>
        <span className="ukpi-sub">Inter-model standard deviation</span>
      </div>

      {/* 4. Confidence / Uncertainty Tier */}
      <div className="uncertainty-kpi-card">
        <div className="ukpi-header">
          <span className="ukpi-label">Confidence Tier</span>
          <div className="ukpi-icon-pill icon-purple">
            <ShieldCheck size={15} />
          </div>
        </div>
        <div className="ukpi-val-row">
          <span className="ukpi-tier-text">{confidence}</span>
        </div>
        <span className="ukpi-sub">Derived from multi-model spread</span>
      </div>
    </div>
  );
}
