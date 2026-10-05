import React from 'react';
import { Activity } from 'lucide-react';

export default function ModelTelemetry({ modelsData }) {
  const defaultModels = [
    { id: 'ecmwf', name: 'ECMWF IFS', type: 'Global NWP', status: 'Online', latency: '48ms' },
    { id: 'graphcast', name: 'GraphCast', type: 'AI Neural', status: 'Online', latency: '12ms' },
    { id: 'pangu', name: 'Pangu-Weather', type: 'AI Transformer', status: 'Online', latency: '14ms' },
    { id: 'gfs', name: 'NOAA GFS', type: 'Global NWP', status: 'Online', latency: '35ms' },
    { id: 'regional_wrf', name: 'Regional WRF', type: 'High-Res WRF', status: 'Online', latency: '62ms' },
  ];

  let models = defaultModels;
  if (modelsData && typeof modelsData === 'object' && Object.keys(modelsData).length > 0) {
    const dataKeys = Object.keys(modelsData);
    const isLiveFeed = dataKeys.includes('open_meteo') || dataKeys.includes('skycast_blend');
    
    if (isLiveFeed) {
      models = dataKeys.map((k) => {
        const item = modelsData[k];
        return {
          id: k,
          name: item?.name || (k === 'open_meteo' ? 'Open-Meteo Feed' : k === 'skycast_blend' ? 'SkyCast Blending' : k === 'emos_engine' ? 'EMOS Calibration' : k.toUpperCase()),
          type: item?.type || 'Operational Pipeline Component',
          status: item?.status || 'Online',
          latency: item?.latency || '35ms',
        };
      });
    } else {
      models = defaultModels.map((m) => {
        const liveSpec = modelsData[m.id];
        return {
          ...m,
          name: liveSpec?.name || m.name,
          type: liveSpec?.type || m.type,
          status: liveSpec?.status || m.status,
          latency: liveSpec?.latency || m.latency,
        };
      });
    }
  }

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'online':
        return { dot: '#22c55e', text: '#15803d', bg: '#f0fdf4', border: '#bbf7d0' };
      case 'processing':
        return { dot: '#3b82f6', text: '#1d4ed8', bg: '#eff6ff', border: '#bfdbfe' };
      case 'delayed':
        return { dot: '#f59e0b', text: '#b45309', bg: '#fffbeb', border: '#fde68a' };
      case 'offline':
        return { dot: '#ef4444', text: '#b91c1c', bg: '#fef2f2', border: '#fecaca' };
      default:
        return { dot: '#94a3b8', text: '#475569', bg: '#f1f5f9', border: '#e2e8f0' };
    }
  };

  return (
    <div className="overview-card telemetry-card-full">
      <div className="overview-card-header">
        <div className="telemetry-header-title">
          <Activity size={16} className="telemetry-activity-icon" />
          <h3 className="overview-card-title">MODEL TELEMETRY</h3>
        </div>
        <span className="telemetry-status-summary">5 / 5 Systems Active</span>
      </div>

      <div className="telemetry-grid">
        {models.map((m) => {
          const colors = getStatusColor(m.status);
          return (
            <div key={m.id} className="telemetry-item">
              <div className="telemetry-item-info">
                <span className="telemetry-model-name">{m.name}</span>
                <span className="telemetry-model-type">{m.type}</span>
              </div>

              <div
                className="telemetry-badge"
                style={{
                  background: colors.bg,
                  color: colors.text,
                  borderColor: colors.border,
                }}
              >
                <span className="telemetry-dot" style={{ background: colors.dot }}></span>
                <span>{m.status}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
