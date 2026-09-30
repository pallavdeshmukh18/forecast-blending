import React, { useState, useEffect, useCallback } from 'react';
import { AlertCircle, RefreshCw, Layers } from 'lucide-react';
import ForecastControls from './ForecastControls';
import BlendedForecastCard from './BlendedForecastCard';
import ModelForecastComparison from './ModelForecastComparison';
import ForecastAgreement from './ForecastAgreement';
import ForecastTimeline from './ForecastTimeline';
import LocationSummary from './LocationSummary';
import ForecastInterpretation from './ForecastInterpretation';
import { getForecastExplorerData, generateFallbackForecastExplorer } from '../../../services/api';

export default function ForecastExplorer() {
  const [location, setLocation] = useState('mumbai');
  const [variable, setVariable] = useState('rainfall');
  const [leadTime, setLeadTime] = useState('24');
  const [run, setRun] = useState('latest');

  const [forecastData, setForecastData] = useState(() =>
    generateFallbackForecastExplorer('mumbai', 'rainfall', 24, 'latest')
  );
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fetchForecast = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getForecastExplorerData({
        location,
        variable,
        leadTime,
        run,
      });

      if (res && res.data) {
        setForecastData(res.data);
      } else {
        setHasError(true);
        setErrorMessage('No forecast data returned for the requested parameters.');
      }
    } catch (err) {
      console.error('[ForecastExplorer] Error loading forecast:', err);
      setHasError(true);
      setErrorMessage(err.message || 'The forecast service could not be reached.');
    } finally {
      setIsLoading(false);
    }
  }, [location, variable, leadTime, run]);

  useEffect(() => {
    fetchForecast();
  }, [fetchForecast]);

  return (
    <div className="forecast-explorer-container">
      {/* 1. Header */}
      <div className="overview-header forecast-explorer-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Forecast Explorer</h1>
          <p className="overview-subtitle">
            Explore blended forecasts across variables, locations and forecast horizons.
          </p>
        </div>

        <div className="overview-header-right">
          <div className="run-telemetry-pill">
            <span className="run-telemetry-label">Latest Run</span>
            <span className="run-telemetry-time">14:00 IST</span>
            <span className="run-telemetry-status">
              <span className="live-status-dot"></span>
              Complete
            </span>
          </div>
        </div>
      </div>

      {/* 2. Controls Bar */}
      <ForecastControls
        selectedLocation={location}
        onLocationChange={setLocation}
        selectedVariable={variable}
        onVariableChange={setVariable}
        selectedLeadTime={leadTime}
        onLeadTimeChange={setLeadTime}
        selectedRun={run}
        onRunChange={setRun}
        isLoading={isLoading}
      />

      {/* 3. Inline Loading Notification */}
      {isLoading && (
        <div className="forecast-inline-loader">
          <RefreshCw size={15} className="spinner-icon" />
          <span>Updating forecast... Retrieving model forecasts</span>
        </div>
      )}

      {/* 4. Error State */}
      {hasError ? (
        <div className="forecast-error-box">
          <AlertCircle size={28} className="error-icon" />
          <div className="error-text-content">
            <h4>Unable to load forecast</h4>
            <p>{errorMessage || 'The forecast service could not be reached.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchForecast}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : !forecastData && !isLoading ? (
        /* 5. Unavailable / Empty State */
        <div className="forecast-empty-box">
          <Layers size={32} className="empty-icon" />
          <h4>Forecast unavailable</h4>
          <p>No forecast data is currently available for this location and lead time. Try another forecast horizon.</p>
        </div>
      ) : (
        /* 6. Active Forecast Explorer Data Sections */
        <div className={`forecast-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Main Hero Blended Forecast Card */}
          <section className="forecast-section">
            <BlendedForecastCard forecastData={forecastData} isLoading={isLoading} />
          </section>

          {/* Model-by-Model Comparison & Model Agreement / Spread */}
          <section className="forecast-section two-column-grid">
            <ModelForecastComparison forecastData={forecastData} />
            <ForecastAgreement forecastData={forecastData} />
          </section>

          {/* Forecast Timeline across horizons */}
          <section className="forecast-section">
            <ForecastTimeline
              forecastData={forecastData}
              selectedLeadTime={leadTime}
              onLeadTimeChange={setLeadTime}
            />
          </section>

          {/* Location Summary & Forecast Interpretation */}
          <section className="forecast-section two-column-grid">
            <LocationSummary forecastData={forecastData} />
            <ForecastInterpretation forecastData={forecastData} />
          </section>
        </div>
      )}
    </div>
  );
}
