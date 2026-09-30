import React from 'react';
import { Cpu, Globe, Database, ShieldCheck, Binary, BarChart3 } from 'lucide-react';

export default function ModelPerformanceCards({ models, unit, periodLabel }) {
  const getModelIcon = (id, isBlend) => {
    if (isBlend) return ShieldCheck;
    if (id === 'graphcast' || id === 'pangu') return Cpu;
    if (id === 'regional_wrf') return Database;
    return Globe;
  };

  return (
    <div className="skill-cards-grid">
      {models.map((model) => {
        const IconComponent = getModelIcon(model.id, model.isBlend);
        const isBlend = !!model.isBlend;

        return (
          <div
            key={model.id}
            className={`skill-model-card ${isBlend ? 'is-blend-card' : ''}`}
          >
            {/* Top Row: Icon + Name + Architecture Badge */}
            <div className="smc-top-row">
              <div className="smc-title-group">
                <div className={`smc-icon-pill ${isBlend ? 'blend-pill' : ''}`}>
                  <IconComponent size={16} strokeWidth={isBlend ? 2.5 : 2} />
                </div>
                <div>
                  <h4 className="smc-model-name">{model.name}</h4>
                  <span className="smc-model-type">{model.type}</span>
                </div>
              </div>

              {isBlend && (
                <div className="smc-blend-tag">
                  <span>Hybrid Blend</span>
                </div>
              )}
            </div>

            {/* Primary Verification Metric: Root Mean Squared Error */}
            <div className="smc-primary-metric-box">
              <div className="smc-primary-header">
                <span className="smc-stat-label">PRIMARY VERIFICATION (RMSE)</span>
                <span className="smc-unit-label">[{unit}]</span>
              </div>
              <div className="smc-primary-val-row">
                <span className={`smc-primary-num ${isBlend ? 'highlight-blend-num' : ''}`}>
                  {model.rmse.toFixed(2)}
                </span>
                <span className="smc-primary-unit">{unit}</span>
              </div>
            </div>

            {/* Secondary Verification Metrics Grid */}
            <div className="smc-secondary-grid">
              <div className="smc-sec-cell">
                <span className="smc-sec-lbl">MAE</span>
                <span className="smc-sec-val">{model.mae.toFixed(2)} {unit}</span>
              </div>

              <div className="smc-sec-cell">
                <span className="smc-sec-lbl">Bias</span>
                <span className={`smc-sec-val ${model.bias > 0 ? 'bias-pos' : model.bias < 0 ? 'bias-neg' : ''}`}>
                  {model.bias > 0 ? `+${model.bias.toFixed(2)}` : model.bias.toFixed(2)}
                </span>
              </div>

              <div className="smc-sec-cell">
                <span className="smc-sec-lbl">Correlation</span>
                <span className="smc-sec-val">{model.corr.toFixed(2)}</span>
              </div>

              <div className="smc-sec-cell">
                <span className="smc-sec-lbl">CRPS</span>
                <span className="smc-sec-val">{model.crps.toFixed(2)}</span>
              </div>
            </div>

            {/* Footer Context Text */}
            <div className="smc-footer-context">
              <span className="smc-context-text">
                {periodLabel || 'Historical verification'}
              </span>
              <span className="smc-context-sub">
                Verified against IMD obs
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
