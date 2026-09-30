import React, { useState, useEffect, useCallback } from 'react';
import { HelpCircle, RefreshCw, AlertTriangle, Database } from 'lucide-react';
import UncertaintyFilters from './UncertaintyFilters';
import ForecastRangeCard from './ForecastRangeCard';
import UncertaintySummaryCards from './UncertaintySummaryCards';
import ModelDisagreement from './ModelDisagreement';
import PredictionIntervalChart from './PredictionIntervalChart';
import RainfallUncertainty from './RainfallUncertainty';
import UpperTailRisk from './UpperTailRisk';
import UncertaintyByHorizon from './UncertaintyByHorizon';
import UncertaintyInterpretation from './UncertaintyInterpretation';
import {
  getUncertaintyData,
  generateFallbackUncertainty,
  INDIAN_LOCATIONS,
} from '../../../services/api';

export default function Uncertainty() {
  const [location, setLocation] = useState('mumbai');
  const [variable, setVariable] = useState('rainfall');
  const [leadTime, setLeadTime] = useState('24');
  const [run, setRun] = useState('latest');

  const [uncertaintyData, setUncertaintyData] = useState(() =>
    generateFallbackUncertainty('mumbai', 'rainfall', 24, 'latest')
  );
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fetchUncertainty = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getUncertaintyData({
        location,
        variable,
        leadTime,
        run,
      });

      if (res && res.data) {
        setUncertaintyData(res.data);
      } else {
        setHasError(true);
        setErrorMessage('Unable to load uncertainty data.');
      }
    } catch (err) {
      console.error('[Uncertainty] Error loading data:', err);
      setHasError(true);
      setErrorMessage(err.message || 'Unable to load uncertainty data.');
    } finally {
      setIsLoading(false);
    }
  }, [location, variable, leadTime, run]);

  useEffect(() => {
    fetchUncertainty();
  }, [fetchUncertainty]);

  const locObj = INDIAN_LOCATIONS.find((l) => l.id === location) || INDIAN_LOCATIONS[0];
  const locName = `${locObj.name}, ${locObj.state}`;

  return (
    <div className="uncertainty-page-container">
      {/* 1. Header */}
      <div className="overview-header uncertainty-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Uncertainty</h1>
          <p className="overview-subtitle">
            Understand forecast confidence, model spread and prediction ranges
          </p>
        </div>

        <div className="overview-header-right">
          <div className="uncertainty-header-telemetry-pill">
            <span className="telemetry-lbl">Last updated:</span>
            <span className="telemetry-time">14:00 IST</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-run">Run #20260930-1400</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-status-live">
              <span className="live-status-dot status-green"></span>
              EMOS / GEV Calibrated
            </span>
          </div>
        </div>
      </div>

      {/* 2. Filter Bar */}
      <UncertaintyFilters
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

      {/* 3. Inline Loading Indicator */}
      {isLoading && (
        <div className="forecast-inline-loader">
          <RefreshCw size={14} className="spinner-icon" />
          <span>Computing Bayesian EMOS prediction intervals and model spread...</span>
        </div>
      )}

      {/* 4. Error State */}
      {hasError ? (
        <div className="forecast-error-box">
          <AlertTriangle size={28} className="error-icon" />
          <div className="error-text-content">
            <h4>Unable to load uncertainty data.</h4>
            <p>{errorMessage || 'The probabilistic forecast service could not be reached.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchUncertainty}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : !uncertaintyData ? (
        /* Empty / No Data State */
        <div className="overview-placeholder-container">
          <div className="placeholder-box">
            <Database size={36} color="#64748b" />
            <h3>Uncertainty information is not available for the selected forecast configuration.</h3>
            <p>Please select a different location, variable, or forecast lead time.</p>
            <button className="placeholder-back-btn" onClick={() => { setVariable('rainfall'); setLeadTime('24'); }}>
              <span>Reset Configuration</span>
            </button>
          </div>
        </div>
      ) : (
        /* 5. Core Operational Uncertainty Content */
        <div className={`uncertainty-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Section 3: Prominent Forecast Range Card */}
          <section className="uncertainty-section">
            <ForecastRangeCard data={uncertaintyData} />
          </section>

          {/* Section 4: Uncertainty Summary Cards (4 KPIs) */}
          <section className="uncertainty-section">
            <UncertaintySummaryCards data={uncertaintyData} />
          </section>

          {/* Section 6: Prediction Interval Time Series Chart */}
          <section className="uncertainty-section">
            <PredictionIntervalChart
              horizonsList={uncertaintyData.horizonsList}
              unit={uncertaintyData.unit}
              varName={uncertaintyData.varName}
            />
          </section>

          {/* Section 5: Model Disagreement & Spread */}
          <section className="uncertainty-section">
            <ModelDisagreement
              models={uncertaintyData.models}
              unit={uncertaintyData.unit}
              modelSpread={uncertaintyData.modelSpread}
            />
          </section>

          {/* Sections 7, 8 & 10: Rainfall Dynamics / Upper-Tail Risk & Interpretation */}
          <section className="uncertainty-section two-column-grid">
            {variable === 'rainfall' ? (
              <RainfallUncertainty
                rainfallSpecific={uncertaintyData.rainfallSpecific}
                centralForecast={uncertaintyData.centralForecast}
                unit={uncertaintyData.unit}
              />
            ) : (
              <UpperTailRisk
                upperTailRisk={uncertaintyData.upperTailRisk}
                unit={uncertaintyData.unit}
                varName={uncertaintyData.varName}
              />
            )}
            <UncertaintyInterpretation
              data={uncertaintyData}
              locationName={locName}
            />
          </section>

          {/* Section 9: Uncertainty Over Forecast Horizons */}
          <section className="uncertainty-section">
            <UncertaintyByHorizon
              horizonsList={uncertaintyData.horizonsList}
              unit={uncertaintyData.unit}
              currentLeadTime={leadTime}
              onSelectHorizon={setLeadTime}
            />
          </section>
        </div>
      )}
    </div>
  );
}
