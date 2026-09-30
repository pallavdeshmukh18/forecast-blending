import React from 'react';
import { ShieldCheck, CheckCircle2, Layers } from 'lucide-react';

export default function BlendSummaryCard({ data }) {
  const blend = data?.blend || { value: 42, confidence: '78%', agreement: 'Moderate' };
  const unit = data?.unit || 'mm';
  const varLabel = data?.variableLabel || 'Rainfall';

  const getAgreementBadge = (agreement) => {
    switch ((agreement || '').toLowerCase()) {
      case 'high':
        return { text: '#059669', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.3)' };
      case 'low':
        return { text: '#dc2626', bg: 'rgba(239, 68, 68, 0.12)', border: 'rgba(239, 68, 68, 0.3)' };
      case 'moderate':
      default:
        return { text: '#d97706', bg: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.3)' };
    }
  };

  const agreementStyle = getAgreementBadge(blend.agreement);

  return (
    <div className="overview-card blend-summary-box-card">
      <div className="blend-summary-header">
        <div className="card-section-tag">OPTIMAL COMPOSITE</div>
        <h3 className="overview-card-title">SKYCAST BLEND</h3>
      </div>

      <div className="blend-summary-center">
        <div className="blend-big-number">
          <span className="blend-num">{blend.value}</span>
          <span className="blend-unit">{unit}</span>
        </div>
        <span className="blend-var-label">{varLabel}</span>
      </div>

      <div className="blend-summary-footer-metrics">
        <div className="blend-metric-chip">
          <span className="chip-label">Confidence:</span>
          <span className="chip-val confidence-val">{blend.confidence}</span>
        </div>

        <div className="blend-metric-chip">
          <span className="chip-label">Agreement:</span>
          <span
            className="chip-val agreement-pill"
            style={{
              color: agreementStyle.text,
              backgroundColor: agreementStyle.bg,
              borderColor: agreementStyle.border,
            }}
          >
            {blend.agreement}
          </span>
        </div>
      </div>
    </div>
  );
}
