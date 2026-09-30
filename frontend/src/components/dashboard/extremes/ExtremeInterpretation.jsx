import React from 'react';
import { FileText, ShieldAlert, AlertTriangle } from 'lucide-react';

export default function ExtremeInterpretation({ selectedHazard, locationName, leadTime }) {
  const prob = selectedHazard?.probability ?? 72;
  const thresh = selectedHazard?.threshold ?? '≥ 50 mm';
  const horizon = leadTime ? `+${String(leadTime).replace('+', '').replace('h', '')}h` : '+24h';
  const varLabel = (selectedHazard?.variable || 'rainfall').toLowerCase();
  const isHazardActive = selectedHazard?.isHazardActive ?? false;

  return (
    <div className="overview-card extreme-interpretation-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">OPERATIONAL GUIDANCE</span>
          <h3 className="overview-card-title">FORECAST INTERPRETATION</h3>
        </div>
        <div className="interp-icon-pill">
          <FileText size={16} color="#0284c7" />
        </div>
      </div>

      <div className="interpretation-body-text">
        <p className="interp-p">
          The selected multi-model blended forecast for <strong>{locationName}</strong> indicates a{' '}
          <strong className="interp-prob-highlight">{prob}% calibrated probability</strong> of exceeding the{' '}
          <strong>{thresh}</strong> {varLabel} threshold at <strong>{horizon}</strong>.
        </p>

        {isHazardActive ? (
          <p className="interp-p interp-alert-callout">
            The forecast probability remains elevated across the next available forecast horizons. Physical NWP and neural model consensus suggests heightened vigilance for localized impacts.
          </p>
        ) : (
          <p className="interp-p interp-nominal-callout">
            Atmospheric parameters remain below critical severe-weather criteria. Standard regional monitoring schedules apply.
          </p>
        )}

        <div className="interp-meta-note">
          <span>* Deterministic guidance derived strictly from calibrated Bayesian EMOS exceedance thresholds and GEV tail modeling.</span>
        </div>
      </div>
    </div>
  );
}
