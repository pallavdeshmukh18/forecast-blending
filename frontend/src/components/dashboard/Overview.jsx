import React, { useState, useEffect, useCallback } from 'react';
import { RefreshCw, AlertCircle } from 'lucide-react';
import OverviewKpiCards from './OverviewKpiCards';
import ForecastSummary from './ForecastSummary';
import WeatherRegimeCard from './WeatherRegimeCard';
import ModelTelemetry from './ModelTelemetry';
import ModelAgreement from './ModelAgreement';
import ActiveHazards from './ActiveHazards';
import LatestForecastRun from './LatestForecastRun';
import {
  checkBackendStatus,
  runForecastPipeline,
  getModelWeights,
  getModels,
  getExtremeAlerts,
  getDiagnostics,
  overviewAdapter,
} from '../../api';

export default function Overview({ onNavigateTab }) {
  const [leadTime, setLeadTime] = useState('24h');
  const [lastUpdated, setLastUpdated] = useState('');
  const [isLiveConnected, setIsLiveConnected] = useState(false);
  const [backendVersion, setBackendVersion] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendNotice, setBackendNotice] = useState(null);

  // Operational state data populated from backend adapter
  const [kpiData, setKpiData] = useState({
    rainfallRisk: { percent: '68%', level: 'High' },
    temperature: { value: '34.8°C', change: '+1.2°C' },
    wind: { value: '42 km/h', level: 'Moderate' },
    confidence: { percent: '87%', status: 'High agreement' },
  });

  const [forecastData, setForecastData] = useState({
    rainfall: { value: '42 mm', change: '↑ 8 mm' },
    temperature: { value: '31.4°C', change: '↓ 0.6°C' },
    wind: { value: '28 km/h', change: '↑ 4 km/h' },
  });

  const [regimeData, setRegimeData] = useState({
    name: 'Heavy Convective Monsoon',
    confidence: '82%',
    description: 'Current atmospheric conditions indicate a heavy-convective regime across the regional domain.',
  });

  const [telemetryData, setTelemetryData] = useState({
    ecmwf: { status: 'Online', latency: '48ms', type: 'Global NWP' },
    graphcast: { status: 'Online', latency: '12ms', type: 'AI Neural' },
    pangu: { status: 'Online', latency: '14ms', type: 'AI Transformer' },
    gfs: { status: 'Online', latency: '35ms', type: 'Global NWP' },
    regional_wrf: { status: 'Online', latency: '62ms', type: 'High-Res WRF' },
  });

  const [weightsData, setWeightsData] = useState({
    weights: [
      { name: 'ECMWF IFS', percent: 34, color: '#0284c7' },
      { name: 'GraphCast', percent: 24, color: '#38bdf8' },
      { name: 'Pangu-Weather', percent: 20, color: '#818cf8' },
      { name: 'NOAA GFS', percent: 14, color: '#f59e0b' },
      { name: 'Regional WRF', percent: 8, color: '#10b981' },
    ],
    agreementStatus: 'Moderate model agreement',
  });

  const [hazardsData, setHazardsData] = useState({
    hazards: [
      { id: 'rain', name: 'Heavy Rain', regions: '3 regions affected', severity: 'Warning', color: '#dc2626', bg: '#fef2f2', border: '#fecaca' },
      { id: 'wind', name: 'High Wind', regions: '2 regions affected', severity: 'Watch', color: '#d97706', bg: '#fffbeb', border: '#fde68a' },
      { id: 'heat', name: 'Heat', regions: '1 region affected', severity: 'Advisory', color: '#ea580c', bg: '#fff7ed', border: '#ffedd5' },
    ],
  });

  const [runData, setRunData] = useState({
    runId: '#20260930-1400',
    initTime: '14:00 IST',
    status: 'Complete',
    sources: '5 / 5',
    blending: 'Complete',
    calibration: 'Complete',
    duration: '42 sec',
  });

  // Calculate current IST timestamp
  const updateTimestamp = () => {
    const now = new Date();
    const istHours = String((now.getUTCHours() + 5 + Math.floor((now.getUTCMinutes() + 30) / 60)) % 24).padStart(2, '0');
    const istMinutes = String((now.getUTCMinutes() + 30) % 60).padStart(2, '0');
    setLastUpdated(`${istHours}:${istMinutes} IST`);
  };

  const fetchOverviewData = useCallback(async () => {
    setIsLoading(true);
    setBackendNotice(null);

    const leadNum = parseInt(leadTime.replace('h', '').replace('+', ''), 10) || 24;

    try {
      // 1. Probe FastAPI health endpoint
      const healthStatus = await checkBackendStatus();
      setIsLiveConnected(healthStatus.isLive);
      if (healthStatus.version) {
        setBackendVersion(healthStatus.version);
      }

      if (!healthStatus.isLive) {
        setBackendNotice('FastAPI backend offline at http://localhost:8000. Displaying cached operational telemetry.');
      }

      // 2. Fetch data in parallel using Promise.allSettled for partial fault tolerance
      const [
        rainForecastRes,
        tempForecastRes,
        windForecastRes,
        weightsRes,
        modelsRes,
        alertsRes,
        diagRes,
      ] = await Promise.allSettled([
        runForecastPipeline({ mode: 'synthetic', variable: 'rainfall', leadTime: leadNum }),
        runForecastPipeline({ mode: 'synthetic', variable: 'temperature', leadTime: leadNum }),
        runForecastPipeline({ mode: 'synthetic', variable: 'wind_speed', leadTime: leadNum }),
        getModelWeights({ variable: 'rainfall', leadTime: leadNum }),
        getModels(),
        getExtremeAlerts(),
        getDiagnostics(),
      ]);

      const rainData = rainForecastRes.status === 'fulfilled' ? rainForecastRes.value : null;
      const tempData = tempForecastRes.status === 'fulfilled' ? tempForecastRes.value : null;
      const windData = windForecastRes.status === 'fulfilled' ? windForecastRes.value : null;
      const weights = weightsRes.status === 'fulfilled' ? weightsRes.value : null;
      const models = modelsRes.status === 'fulfilled' ? modelsRes.value : null;
      const alerts = alertsRes.status === 'fulfilled' ? alertsRes.value : null;
      const diag = diagRes.status === 'fulfilled' ? diagRes.value : null;

      // 3. Adapt and map backend responses to UI
      if (rainData || tempData || windData || weights || alerts) {
        setKpiData(
          overviewAdapter.adaptKpis({
            rainForecast: rainData,
            tempForecast: tempData,
            windForecast: windData,
            weightsData: weights,
            alertsData: alerts,
          })
        );

        setForecastData(
          overviewAdapter.adaptForecastSummary({
            rainForecast: rainData,
            tempForecast: tempData,
            windForecast: windData,
          })
        );

        if (rainData?.regime) {
          setRegimeData(overviewAdapter.adaptWeatherRegime({ forecastRes: rainData }));
        }

        if (models) {
          setTelemetryData(
            overviewAdapter.adaptModelTelemetry({
              modelsMeta: models,
              isLiveConnected: healthStatus.isLive,
            })
          );
        }

        if (weights) {
          setWeightsData(overviewAdapter.adaptModelAgreement({ weightsRes: weights }));
        }

        if (alerts) {
          setHazardsData(overviewAdapter.adaptActiveHazards({ alertsRes: alerts }));
        }

        if (diag || rainData) {
          setRunData(
            overviewAdapter.adaptLatestRun({
              diagRes: diag,
              forecastRes: rainData,
            })
          );
        }
      }
    } catch (err) {
      console.warn('[Overview] Error loading data:', err);
      setBackendNotice('Unable to reach operational FastAPI service. Showing cached state.');
    } finally {
      setIsLoading(false);
      updateTimestamp();
    }
  }, [leadTime]);

  useEffect(() => {
    fetchOverviewData();
  }, [fetchOverviewData]);

  return (
    <div className="overview-page-container">
      {/* 1. Dashboard Header */}
      <div className="overview-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Overview</h1>
          <p className="overview-subtitle">
            Current forecast intelligence and system status
          </p>
        </div>

        <div className="overview-header-right">
          <div className="last-updated-pill">
            <span className="last-updated-label">Last updated:</span>
            <span className="last-updated-time">{lastUpdated || '14:00 IST'}</span>
          </div>

          <div className={`live-status-indicator ${isLiveConnected ? 'connected' : 'standby'}`}>
            <span className="live-status-dot"></span>
            <span>
              {isLiveConnected
                ? `FastAPI Live ${backendVersion ? `v${backendVersion}` : ''}`
                : 'Backend Standby (Offline)'}
            </span>
          </div>

          <button
            type="button"
            className="overview-refresh-btn"
            onClick={fetchOverviewData}
            disabled={isLoading}
            title="Refresh operational data from FastAPI backend"
          >
            <RefreshCw size={13} className={isLoading ? 'spinner-icon' : ''} />
            <span>Sync</span>
          </button>
        </div>
      </div>

      {/* Backend Offline / Status Banner */}
      {backendNotice && (
        <div className="overview-backend-banner">
          <AlertCircle size={15} className="banner-icon" />
          <span>{backendNotice}</span>
          <button type="button" className="banner-retry-btn" onClick={fetchOverviewData}>
            Retry Connection
          </button>
        </div>
      )}

      {/* 2. Four KPI Cards */}
      <section className={`overview-section ${isLoading ? 'is-syncing' : ''}`}>
        <OverviewKpiCards data={kpiData} />
      </section>

      {/* 3. Forecast Summary + Active Weather Regime (Two Columns) */}
      <section className={`overview-section two-column-grid ${isLoading ? 'is-syncing' : ''}`}>
        <ForecastSummary
          leadTime={leadTime}
          onLeadTimeChange={setLeadTime}
          data={forecastData}
        />
        <WeatherRegimeCard regimeData={regimeData} />
      </section>

      {/* 4. Model Status / Telemetry (Full Width) */}
      <section className={`overview-section ${isLoading ? 'is-syncing' : ''}`}>
        <ModelTelemetry modelsData={telemetryData} />
      </section>

      {/* 5. Model Agreement + Active Hazards (Two Columns) */}
      <section className={`overview-section two-column-grid ${isLoading ? 'is-syncing' : ''}`}>
        <ModelAgreement weightsData={weightsData} />
        <ActiveHazards
          hazardsData={hazardsData}
          onNavigateToExtremes={() => onNavigateTab && onNavigateTab('extremes')}
        />
      </section>

      {/* 6. Latest Forecast Run (Full Width Operational Status Card) */}
      <section className={`overview-section ${isLoading ? 'is-syncing' : ''}`}>
        <LatestForecastRun
          runData={runData}
          onNavigateToRuns={() => onNavigateTab && onNavigateTab('runs')}
        />
      </section>
    </div>
  );
}
