import React from 'react';
import { MapPin, Compass, Shield, GitCommit } from 'lucide-react';

export default function LocationSummary({ forecastData }) {
  const loc = forecastData?.location;
  const locName = loc ? `${loc.name}, ${loc.state}` : 'Mumbai, Maharashtra';
  const leadHorizon = forecastData?.leadTime ? `+${String(forecastData.leadTime).replace('+', '').replace('h', '')} hours` : '+24 hours';
  const regime = forecastData?.regime || 'Heavy Convective Monsoon';
  const confidence = forecastData?.spread?.confidence || 'Confidence data unavailable';
  const agreement = forecastData?.spread?.agreement || 'Moderate';

  return (
    <div className="overview-card location-summary-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">GEOGRAPHIC METRICS</span>
          <h3 className="overview-card-title">LOCATION SUMMARY</h3>
        </div>
        <div className="loc-summary-pin">
          <MapPin size={16} color="#0284c7" />
        </div>
      </div>

      <div className="location-summary-header-city">
        <h4 className="city-title">{locName}</h4>
        {loc?.region && <span className="city-geo-zone">{loc.region.replace('_', ' ').toUpperCase()} REGION</span>}
      </div>

      <div className="location-summary-specs">
        <div className="summary-spec-row">
          <span className="spec-label">
            <GitCommit size={13} className="spec-icon" />
            <span>Forecast horizon</span>
          </span>
          <span className="spec-val">{leadHorizon}</span>
        </div>

        <div className="summary-spec-row">
          <span className="spec-label">
            <Compass size={13} className="spec-icon" />
            <span>Active regime</span>
          </span>
          <span className="spec-val highlight-regime">{regime}</span>
        </div>

        <div className="summary-spec-row">
          <span className="spec-label">
            <Shield size={13} className="spec-icon" />
            <span>Forecast confidence</span>
          </span>
          <span className="spec-val">{confidence}</span>
        </div>

        <div className="summary-spec-row">
          <span className="spec-label">
            <GitCommit size={13} className="spec-icon" />
            <span>Model agreement</span>
          </span>
          <span className="spec-val">{agreement}</span>
        </div>
      </div>
    </div>
  );
}
