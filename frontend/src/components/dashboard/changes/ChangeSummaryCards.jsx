import React from 'react';
import { ArrowUpRight, ArrowDownRight, ArrowRight, GitCommit, Clock } from 'lucide-react';

export default function ChangeSummaryCards({ activeVarChange, runMeta }) {
  if (!activeVarChange) return null;

  const { previous, current, change, changePct, unit, name, direction } = activeVarChange;
  const isUp = change > 0;
  const isDown = change < 0;
  const isZero = change === 0;

  const DirectionIcon = isUp ? ArrowUpRight : isDown ? ArrowDownRight : ArrowRight;

  return (
    <div className="forecast-change-summary-grid">
      {/* 1. Previous Run Forecast */}
      <div className="change-kpi-card">
        <div className="ckpi-header">
          <span className="ckpi-lbl">Previous Forecast Run</span>
          <span className="ckpi-run-tag">{runMeta?.compareRun?.timeIST || '08:00 IST'}</span>
        </div>
        <div className="ckpi-val-row">
          <span className="ckpi-num muted">{previous.toFixed(1)}</span>
          <span className="ckpi-unit">{unit}</span>
        </div>
        <span className="ckpi-sub">{runMeta?.compareRun?.label || 'Baseline run'}</span>
      </div>

      {/* 2. Current Run Forecast */}
      <div className="change-kpi-card highlight-current">
        <div className="ckpi-header">
          <span className="ckpi-lbl">Current Forecast Run</span>
          <span className="ckpi-run-tag live-tag">{runMeta?.currentRun?.timeIST || '14:00 IST'}</span>
        </div>
        <div className="ckpi-val-row">
          <span className="ckpi-num active-color">{current.toFixed(1)}</span>
          <span className="ckpi-unit">{unit}</span>
        </div>
        <span className="ckpi-sub">{runMeta?.currentRun?.label || 'Active blended run'}</span>
      </div>

      {/* 3. Absolute & Percentage Delta */}
      <div className="change-kpi-card delta-card">
        <div className="ckpi-header">
          <span className="ckpi-lbl">Net Run Departure (Δ)</span>
          <div className={`ckpi-direction-pill ${isUp ? 'pill-up' : isDown ? 'pill-down' : 'pill-neutral'}`}>
            <DirectionIcon size={14} strokeWidth={2.5} />
            <span>{isUp ? 'Increase' : isDown ? 'Decrease' : 'Stable'}</span>
          </div>
        </div>
        <div className="ckpi-val-row">
          <span className={`ckpi-num ${isUp ? 'num-up' : isDown ? 'num-down' : ''}`}>
            {change > 0 ? `+${change.toFixed(1)}` : change.toFixed(1)}
          </span>
          <span className="ckpi-unit">{unit}</span>
          <span className="ckpi-pct-badge">
            {changePct > 0 ? `+${changePct.toFixed(1)}%` : `${changePct.toFixed(1)}%`}
          </span>
        </div>
        <span className="ckpi-sub">{activeVarChange.status || 'Inter-run synoptic adjustment'}</span>
      </div>
    </div>
  );
}
