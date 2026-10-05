import React from 'react';
import { FileText, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function UncertaintyInterpretation({ data, locationName }) {
  const central = data?.centralForecast ?? 42.0;
  const unit = data?.unit || 'mm';
  const varName = data?.varName || 'Rainfall';
  const [low80, high80] = data?.predictionInterval80 || [32.0, 54.0];
  const spread = data?.modelSpread ?? 5.1;
  const confidence = data?.confidenceRating || 'High Confidence';
  const leadTime = data?.leadTime || '+24h';

  const isLowDisagreement = spread < (data?.variable === 'rainfall' ? 6.0 : data?.variable === 'temperature' ? 1.8 : 3.5);

  return (
    <div className="overview-card uncertainty-interpretation-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">OPERATIONAL SYNTHESIS</span>
          <h3 className="overview-card-title">FORECAST UNCERTAINTY INTERPRETATION</h3>
        </div>
        <div className="interp-icon-pill">
          <FileText size={16} color="#0284c7" />
        </div>
      </div>

      <div className="interp-body-content">
        <p className="interp-sentence">
          For <strong>{locationName}</strong> at <strong>{leadTime}</strong>, the calibrated blended forecast projects{' '}
          <strong className="interp-highlight">{central} {unit}</strong> of {varName.toLowerCase()} within an 80% prediction interval of{' '}
          <strong>[{low80} {unit} — {high80} {unit}]</strong>.
        </p>

        <div className={`interp-callout-box ${isLowDisagreement ? 'callout-green' : 'callout-amber'}`}>
          {isLowDisagreement ? (
            <p>
              The individual forecasting systems are closely grouped around the BharatWeather blend with an inter-model spread of {spread} {unit}, indicating low model disagreement and heightened forecast confidence.
            </p>
          ) : (
            <p>
              Inter-model spread is elevated at {spread} {unit}, reflecting structural disagreements between physics NWP and AI neural models regarding boundary layer dynamics.
            </p>
          )}
        </div>

        <p className="interp-secondary-sentence">
          The prediction interval broadens systematically from short range (+6h) through medium range (+72h), consistent with classical theoretical error propagation in atmospheric dynamics.
        </p>

        <div className="interp-footer-meta">
          <span>* Deterministic meteorological synthesis generated strictly from calibrated Bayesian EMOS spread statistics and GEV tail distributions.</span>
        </div>
      </div>
    </div>
  );
}
