import React from 'react';
import { CheckCircle2, Clock, AlertTriangle, ArrowRight } from 'lucide-react';

export default function PipelineProgressTracker({ stages = [] }) {
  if (!stages || stages.length === 0) return null;

  return (
    <div className="pipeline-tracker-container">
      <div className="pipeline-stages-flow">
        {stages.map((stage, idx) => {
          const isCompleted = stage.status === 'Completed';
          const isWarning = stage.status === 'Warning';
          const isPending = stage.status === 'Pending';
          const isLast = idx === stages.length - 1;

          return (
            <React.Fragment key={stage.name}>
              <div
                className={`pipeline-stage-node ${
                  isCompleted ? 'node-completed' : isWarning ? 'node-warning' : 'node-pending'
                }`}
              >
                <div className="stage-icon-wrap">
                  {isCompleted && <CheckCircle2 size={14} className="text-emerald-400" />}
                  {isWarning && <AlertTriangle size={14} className="text-amber-400" />}
                  {isPending && <Clock size={14} className="text-slate-400" />}
                </div>
                <div className="stage-text-block">
                  <div className="stage-title">{stage.name}</div>
                  <div className="stage-meta-row">
                    <span className={`stage-status-text ${isCompleted ? 'text-green' : isWarning ? 'text-amber' : ''}`}>
                      {stage.status}
                    </span>
                    {stage.durationSec != null && (
                      <span className="stage-duration font-mono">{stage.durationSec}s</span>
                    )}
                  </div>
                </div>
              </div>

              {!isLast && (
                <div className="pipeline-stage-connector">
                  <ArrowRight size={13} className="connector-arrow" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
