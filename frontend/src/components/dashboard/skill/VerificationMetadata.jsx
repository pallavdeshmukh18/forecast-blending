import React from 'react';
import { Database, Calendar, Layers, Clock, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function VerificationMetadata({ metadata }) {
  const meta = metadata || {
    evaluationPeriod: 'Recent 30 Days (Rolling Window)',
    sampleCount: '1,280 matched station pairs',
    referenceDataset: 'IMD High-Resolution Surface Telemetry & ERA5 Reanalysis',
    initializationRuns: '00:00 & 12:00 UTC operational runs',
    lastUpdate: '14:00 IST today',
    validationMethodology: 'Walk-forward rolling window validation (15-step window, 6-step min train)',
  };

  return (
    <div className="overview-card verification-metadata-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">AUDIT TRAIL</span>
          <h3 className="overview-card-title">VERIFICATION METHODOLOGY & METADATA</h3>
        </div>
        <div className="meta-icon-pill">
          <Database size={16} color="#0284c7" />
        </div>
      </div>

      <div className="verification-meta-grid">
        <div className="meta-item-box">
          <div className="meta-item-header">
            <Calendar size={14} className="meta-icon" />
            <span className="meta-item-lbl">Evaluation Period</span>
          </div>
          <span className="meta-item-val">{meta.evaluationPeriod}</span>
        </div>

        <div className="meta-item-box">
          <div className="meta-item-header">
            <Layers size={14} className="meta-icon" />
            <span className="meta-item-lbl">Forecast Samples</span>
          </div>
          <span className="meta-item-val">{meta.sampleCount}</span>
        </div>

        <div className="meta-item-box">
          <div className="meta-item-header">
            <Database size={14} className="meta-icon" />
            <span className="meta-item-lbl">Reference Ground Truth</span>
          </div>
          <span className="meta-item-val">{meta.referenceDataset}</span>
        </div>

        <div className="meta-item-box">
          <div className="meta-item-header">
            <Clock size={14} className="meta-icon" />
            <span className="meta-item-lbl">Initialization Cycle</span>
          </div>
          <span className="meta-item-val">{meta.initializationRuns}</span>
        </div>

        <div className="meta-item-box full-width">
          <div className="meta-item-header">
            <ShieldCheck size={14} className="meta-icon" />
            <span className="meta-item-lbl">Validation Protocol</span>
          </div>
          <span className="meta-item-val">{meta.validationMethodology}</span>
        </div>
      </div>
    </div>
  );
}
