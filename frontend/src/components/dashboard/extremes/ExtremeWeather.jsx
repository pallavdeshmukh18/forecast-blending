import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, RefreshCw, Layers, ShieldAlert, Clock } from 'lucide-react';
import ExtremeFilters from './ExtremeFilters';
import ActiveHazardBanner from './ActiveHazardBanner';
import HazardOverviewCards from './HazardOverviewCards';
import EventProbabilityGauge from './EventProbabilityGauge';
import EventThresholdCard from './EventThresholdCard';
import ForecastSeverityCard from './ForecastSeverityCard';
import CalibrationUncertaintyCard from './CalibrationUncertaintyCard';
import ExtremeModelAgreement from './ExtremeModelAgreement';
import HazardTimeline from './HazardTimeline';
import ExtremeInterpretation from './ExtremeInterpretation';
import {
  getExtremeWeatherData,
  generateFallbackExtremeWeather,
  INDIAN_LOCATIONS,
} from '../../../services/api';

export default function ExtremeWeather() {
  const [location, setLocation] = useState('mumbai');
  const [hazard, setHazard] = useState('heavy_rain');
  const [leadTime, setLeadTime] = useState('24');
  const [run, setRun] = useState('latest');
  const [selectedThreshold, setSelectedThreshold] = useState(null);

  const [extremeData, setExtremeData] = useState(() =>
    generateFallbackExtremeWeather('mumbai', 'heavy_rain', 24, null)
  );
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fetchExtremeData = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getExtremeWeatherData({
        location,
        hazard,
        leadTime,
        threshold: selectedThreshold,
      });

      if (res && res.data) {
        setExtremeData(res.data);
      } else {
        setHasError(true);
        setErrorMessage('Unable to load extreme weather data.');
      }
    } catch (err) {
      console.error('[ExtremeWeather] Error loading data:', err);
      setHasError(true);
      setErrorMessage(err.message || 'Unable to load extreme weather data.');
    } finally {
      setIsLoading(false);
    }
  }, [location, hazard, leadTime, selectedThreshold]);

  useEffect(() => {
    fetchExtremeData();
  }, [fetchExtremeData]);

  // When user switches hazard card, reset custom threshold to default for that hazard
  const handleSelectHazard = (newHazardId) => {
    setHazard(newHazardId);
    setSelectedThreshold(null);
  };

  const locObj = INDIAN_LOCATIONS.find((l) => l.id === location) || INDIAN_LOCATIONS[0];
  const locName = `${locObj.name}, ${locObj.state}`;

  return (
    <div className="extreme-weather-container">
      {/* 1. Header */}
      <div className="overview-header extreme-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Extreme Weather</h1>
          <p className="overview-subtitle">
            Monitor calibrated weather hazard probabilities and event thresholds
          </p>
        </div>

        <div className="overview-header-right">
          <div className="extreme-header-telemetry-pill">
            <span className="telemetry-lbl">Last updated:</span>
            <span className="telemetry-time">14:00 IST</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-run">Run #20260930-1400</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-status-live">
              <span className="live-status-dot status-green"></span>
              Operational
            </span>
          </div>
        </div>
      </div>

      {/* 2. Filter Bar */}
      <ExtremeFilters
        selectedLocation={location}
        onLocationChange={setLocation}
        selectedHazard={hazard}
        onHazardChange={handleSelectHazard}
        selectedLeadTime={leadTime}
        onLeadTimeChange={setLeadTime}
        selectedRun={run}
        onRunChange={setRun}
        isLoading={isLoading}
      />

      {/* 3. Inline Loading Notification */}
      {isLoading && (
        <div className="forecast-inline-loader">
          <RefreshCw size={14} className="spinner-icon" />
          <span>Updating calibrated hazard probabilities and GEV tail estimates...</span>
        </div>
      )}

      {/* 4. Error State */}
      {hasError ? (
        <div className="forecast-error-box">
          <AlertTriangle size={28} className="error-icon" />
          <div className="error-text-content">
            <h4>Unable to load extreme weather data.</h4>
            <p>{errorMessage || 'The hazard forecasting service could not be reached.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchExtremeData}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : (
        /* 5. Core Operational Hazard Sections */
        <div className={`extreme-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Active Hazard Banner / No Hazard State */}
          <section className="extreme-section">
            <ActiveHazardBanner
              selectedHazard={extremeData?.selectedHazard}
              locationName={locName}
              leadTime={leadTime}
            />
          </section>

          {/* Three Hazard Overview Cards (Heavy Rain, Heatwave, High Wind) */}
          <section className="extreme-section">
            <HazardOverviewCards
              cards={extremeData?.hazardOverviewCards || []}
              selectedHazardId={hazard}
              onSelectHazard={handleSelectHazard}
            />
          </section>

          {/* Event Probability Gauge & Event Threshold */}
          <section className="extreme-section two-column-grid">
            <EventProbabilityGauge
              probability={extremeData?.selectedHazard?.probability}
              threshold={extremeData?.selectedHazard?.threshold}
              hazardName={extremeData?.selectedHazard?.hazardName || 'Extreme Event'}
              calibrationMethod={extremeData?.selectedHazard?.calibrationMethod}
            />
            <EventThresholdCard
              thresholds={extremeData?.selectedHazard?.thresholds}
              currentThresholdValue={extremeData?.selectedHazard?.thresholdValue}
              onSelectThreshold={setSelectedThreshold}
              hazardName={extremeData?.selectedHazard?.hazardName || 'Extreme Event'}
              unit={extremeData?.selectedHazard?.unit}
            />
          </section>

          {/* Forecast Severity & Calibration / Uncertainty */}
          <section className="extreme-section two-column-grid">
            <ForecastSeverityCard selectedHazard={extremeData?.selectedHazard} />
            <CalibrationUncertaintyCard selectedHazard={extremeData?.selectedHazard} />
          </section>

          {/* Model Agreement & Operational Interpretation */}
          <section className="extreme-section two-column-grid">
            <ExtremeModelAgreement
              models={extremeData?.selectedHazard?.models}
              unit={extremeData?.selectedHazard?.unit}
            />
            <ExtremeInterpretation
              selectedHazard={extremeData?.selectedHazard}
              locationName={locName}
              leadTime={leadTime}
            />
          </section>

          {/* Full-width Hazard Timeline across horizons */}
          <section className="extreme-section">
            <HazardTimeline
              timeline={extremeData?.selectedHazard?.timeline}
              currentLeadTime={leadTime}
              onSelectHorizon={setLeadTime}
              unit={extremeData?.selectedHazard?.unit}
            />
          </section>
        </div>
      )}
    </div>
  );
}
