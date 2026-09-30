import React from 'react';
import { Server, Cpu, Database, Layers, Radio, Globe, CheckCircle2 } from 'lucide-react';

const serviceIcons = {
  fastapi_backend: Server,
  forecast_engine: Layers,
  ml_services: Cpu,
  data_pipeline: Database,
  spatial_cnn: Radio,
  frontend_client: Globe,
};

export default function ServiceStatusGrid({ services = [] }) {
  if (!services || services.length === 0) return null;

  return (
    <div className="forecast-explorer-card service-grid-card">
      <div className="card-top-header">
        <div>
          <h3 className="card-heading">Core System Services</h3>
          <p className="card-subheading">Operational health, network latency, and version telemetry across pipeline modules</p>
        </div>
      </div>

      <div className="service-cards-grid">
        {services.map((svc) => {
          const Icon = serviceIcons[svc.id] || Server;
          const isOk = svc.status === 'Operational';

          return (
            <div key={svc.id} className="service-kpi-item">
              <div className="service-item-top">
                <div className="service-icon-box">
                  <Icon size={16} className="text-blue-400" />
                </div>
                <div className="service-meta-head">
                  <h4 className="service-name">{svc.name}</h4>
                  <span className="service-version font-mono">{svc.version}</span>
                </div>
                <span className={`status-badge-compact ${isOk ? 'badge-completed' : 'badge-partial'}`}>
                  <span className={`status-indicator-dot ${isOk ? 'dot-green' : 'dot-amber'}`}></span>
                  {svc.status}
                </span>
              </div>

              <p className="service-desc">{svc.description}</p>

              <div className="service-metrics-row">
                <div className="service-metric">
                  <span className="metric-lbl">Response Time</span>
                  <span className="metric-val font-mono">{svc.latency}</span>
                </div>
                <div className="service-metric">
                  <span className="metric-lbl">Last Probed</span>
                  <span className="metric-val font-mono text-secondary">{svc.lastChecked}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
