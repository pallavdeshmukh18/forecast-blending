import React from 'react';
import { AlertOctagon, TrendingUp, ShieldAlert, Binary } from 'lucide-react';

export default function UpperTailRisk({ upperTailRisk, unit, varName }) {
  if (!upperTailRisk) return null;

  const { p95Estimate, p99Estimate, gevShapeParameter, tailProbability, tailDescription } = upperTailRisk;

  return (
    <div className="overview-card upper-tail-risk-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">EXTREME VALUE THEORY (EVT)</span>
          <h3 className="overview-card-title">UPPER-TAIL RISK & GEV CALIBRATION</h3>
        </div>
        <div className="tail-icon-pill">
          <AlertOctagon size={16} color="#dc2626" />
        </div>
      </div>

      <div className="tail-risk-grid">
        {/* P95 Upper Quantile */}
        <div className="tail-stat-box">
          <span className="tail-stat-lbl">95th Percentile Estimate (P95)</span>
          <div className="tail-val-cluster">
            <span className="tail-num highlight-red">{p95Estimate}</span>
            <span className="tail-unit">{unit}</span>
          </div>
          <span className="tail-sub">1-in-20 event upper tail</span>
        </div>

        {/* P99 Extreme Percentile */}
        <div className="tail-stat-box">
          <span className="tail-stat-lbl">99th Percentile Estimate (P99)</span>
          <div className="tail-val-cluster">
            <span className="tail-num highlight-red">{p99Estimate}</span>
            <span className="tail-unit">{unit}</span>
          </div>
          <span className="tail-sub">1-in-100 extreme synoptic limit</span>
        </div>

        {/* GEV Tail Probability */}
        <div className="tail-stat-box">
          <span className="tail-stat-lbl">Tail Exceedance Probability</span>
          <div className="tail-val-cluster">
            <span className="tail-text-num">{tailProbability}</span>
          </div>
          <span className="tail-sub">GEV calibrated tail frequency</span>
        </div>

        {/* GEV Shape Parameter */}
        <div className="tail-stat-box">
          <span className="tail-stat-lbl">GEV Shape Parameter ($\xi$)</span>
          <div className="tail-val-cluster">
            <span className="tail-text-num monospace">
              {gevShapeParameter > 0 ? `+${gevShapeParameter}` : gevShapeParameter}
            </span>
          </div>
          <span className="tail-sub">{tailDescription}</span>
        </div>
      </div>
    </div>
  );
}
