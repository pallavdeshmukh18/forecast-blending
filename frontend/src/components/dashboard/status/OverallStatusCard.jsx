import React from 'react';
import { ShieldCheck, Activity, Clock, Server } from 'lucide-react';

export default function OverallStatusCard({ overall }) {
  if (!overall) return null;

  const isOperational = overall.statusCode === 'operational';
  const isDegraded = overall.statusCode === 'degraded';

  return (
    <div className={`forecast-explorer-card overall-status-card ${isOperational ? 'status-card-healthy' : 'status-card-warning'}`}>
      <div className="overall-status-header">
        <div className="status-indicator-badge-large">
          <span className={`status-pulse-dot ${isOperational ? 'pulse-green' : 'pulse-amber'}`}></span>
          <span className="status-large-text">{overall.status}</span>
        </div>
        <div className="status-quick-pills">
          <div className="status-quick-pill">
            <Activity size={13} className="text-emerald-400" />
            <span>Uptime: <strong className="font-mono text-white">{overall.uptime}</strong></span>
          </div>
          <div className="status-quick-pill">
            <Server size={13} className="text-blue-400" />
            <span>Monitored Services: <strong className="font-mono text-white">{overall.healthyServices} / {overall.totalServices}</strong></span>
          </div>
          <div className="status-quick-pill">
            <Clock size={13} className="text-secondary" />
            <span>Health Check: <strong className="text-white">{overall.lastHealthCheck}</strong></span>
          </div>
        </div>
      </div>

      <p className="overall-headline-text">
        {overall.headline}
      </p>
    </div>
  );
}
