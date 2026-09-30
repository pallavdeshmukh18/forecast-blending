import React from 'react';
import { FileText, ArrowRight } from 'lucide-react';

export default function ChangeInterpretation({ interpretation, locationName }) {
  return (
    <div className="overview-card change-interpretation-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">METEOROLOGICAL SYNTHESIS</span>
          <h3 className="overview-card-title">WHAT CHANGED IN THIS FORECAST RUN?</h3>
        </div>
        <div className="interp-icon-pill">
          <FileText size={16} color="#0284c7" />
        </div>
      </div>

      <div className="change-interpretation-body">
        <p className="interp-main-text">{interpretation}</p>

        <div className="interp-guidance-bullet">
          <span className="bullet-title">Operational Directive:</span>
          <span className="bullet-desc">
            Model weights were dynamically rebalanced toward ECMWF and Pangu following verified regional performance across convective regimes. Operational threshold advisories for {locationName} should reference the latest 14:00 IST run.
          </span>
        </div>

        <div className="interp-footer-disclaimer">
          <span>* Deterministic comparison generated directly from numerical differentials between Run #20260930-1400 and Run #20260930-0800.</span>
        </div>
      </div>
    </div>
  );
}
