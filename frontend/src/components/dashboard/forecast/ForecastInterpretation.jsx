import React from 'react';
import { FileText, Info } from 'lucide-react';

export default function ForecastInterpretation({ forecastData }) {
  const loc = forecastData?.location;
  const locName = loc ? `${loc.name}, ${loc.state}` : 'the selected location';
  const varLabel = (forecastData?.variableLabel || 'rainfall').toLowerCase();
  const unit = forecastData?.unit || 'mm';
  const blendVal = forecastData?.blendedValue ?? 42;
  const spread = forecastData?.spread || { min: 37, max: 51, agreement: 'Moderate' };
  const minVal = spread.min ?? 37;
  const maxVal = spread.max ?? 51;
  const agreement = (spread.agreement || 'moderate').toLowerCase();

  // Deterministic interpretation sentence based strictly on data values
  const getUncertaintyContext = (agreementStatus) => {
    switch (agreementStatus) {
      case 'high':
        return 'tight consensus and minimal dispersion across numerical prediction and neural models.';
      case 'low':
        return 'elevated disagreement between physics-based NWP systems and AI weather models, warranting operational caution.';
      case 'moderate':
      default:
        return 'some uncertainty between global and regional forecast sources.';
    }
  };

  return (
    <div className="overview-card forecast-interpretation-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">METEOROLOGICAL SYNTHESIS</span>
          <h3 className="overview-card-title">FORECAST SUMMARY</h3>
        </div>
        <div className="interpretation-icon">
          <FileText size={16} color="#3b82f6" />
        </div>
      </div>

      <div className="interpretation-content">
        <p className="interpretation-paragraph">
          The five forecast systems predict {varLabel} between{' '}
          <strong>
            {minVal} {unit}
          </strong>{' '}
          and{' '}
          <strong>
            {maxVal} {unit}
          </strong>{' '}
          for {locName}.
        </p>

        <p className="interpretation-paragraph highlight-blend-statement">
          The BharatWeather blended forecast is{' '}
          <strong className="blend-highlight">
            {blendVal} {unit}
          </strong>
          .
        </p>

        <p className="interpretation-paragraph">
          Model agreement is <strong>{agreement}</strong>, indicating{' '}
          {getUncertaintyContext(agreement)}
        </p>

        <div className="interpretation-footnote">
          <Info size={13} className="info-icon" />
          <span>
            Computed via dynamic regime-conditioned convex weighting based on localized historical verification.
          </span>
        </div>
      </div>
    </div>
  );
}
