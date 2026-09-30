import React from 'react';
import { CheckCircle2, HelpCircle, Sparkles } from 'lucide-react';

export default function AttributionExplainabilityCard({ factors }) {
  const items = factors || [
    {
      id: 'regime',
      label: 'Current weather regime',
      value: 'Heavy Convective Monsoon',
      description: 'Atmospheric stability and convection favor models with calibrated regional parameterization.',
    },
    {
      id: 'leadTime',
      label: 'Lead time',
      value: '+24 hours',
      description: 'Optimal trade-off horizon balancing neural diffusion sharpness and physics-based boundary conditions.',
    },
    {
      id: 'skill',
      label: 'Historical skill',
      value: 'Regional rainfall performance',
      description: 'Calculated using inverse-variance weighting on 30-day verified ground station observations.',
    },
    {
      id: 'recentError',
      label: 'Recent model error',
      value: 'Latest rolling performance',
      description: 'Adaptive penalty applied to systems exhibiting systematic bias over recent forecast cycles.',
    },
  ];

  return (
    <div className="overview-card attribution-explainability-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">ATTRIBUTION LOGIC</span>
          <h3 className="overview-card-title">WHY THESE WEIGHTS?</h3>
        </div>
        <div className="attribution-header-icon">
          <Sparkles size={16} color="#0284c7" />
        </div>
      </div>

      <div className="attribution-items-list">
        {items.map((factor) => (
          <div key={factor.id} className="attribution-item-row">
            <div className="attribution-check-icon">
              <CheckCircle2 size={18} color="#059669" strokeWidth={2.5} />
            </div>

            <div className="attribution-body">
              <span className="attribution-factor-label">{factor.label}</span>
              <h4 className="attribution-factor-val">{factor.value}</h4>
              {factor.description && (
                <p className="attribution-factor-desc">{factor.description}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
