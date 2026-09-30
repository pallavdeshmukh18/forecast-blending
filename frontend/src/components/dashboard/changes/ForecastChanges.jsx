import React, { useState, useEffect, useCallback } from 'react';
import { GitCompare, RefreshCw, AlertTriangle, Database } from 'lucide-react';
import ForecastChangesFilters from './ForecastChangesFilters';
import ChangeSummaryCards from './ChangeSummaryCards';
import ChangeByVariable from './ChangeByVariable';
import ChangeByHorizon from './ChangeByHorizon';
import ModelByModelChange from './ModelByModelChange';
import SkyCastBlendChange from './SkyCastBlendChange';
import UncertaintyAndRegimeChange from './UncertaintyAndRegimeChange';
import SignificantChanges from './SignificantChanges';
import ChangeInterpretation from './ChangeInterpretation';
import {
  getForecastChangesData,
  generateFallbackForecastChanges,
  INDIAN_LOCATIONS,
} from '../../../services/api';

export default function ForecastChanges() {
  const [location, setLocation] = useState('mumbai');
  const [variable, setVariable] = useState('rainfall');
  const [leadTime, setLeadTime] = useState('24');
  const [currentRun, setCurrentRun] = useState('run-1400');
  const [compareRun, setCompareRun] = useState('run-0800');

  const [changesData, setChangesData] = useState(() =>
    generateFallbackForecastChanges('mumbai', 'rainfall', 24, 'run-1400', 'run-0800')
  );
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fetchChangesData = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getForecastChangesData({
        location,
        variable,
        leadTime,
        currentRun,
        compareRun,
      });

      if (res && res.data) {
        setChangesData(res.data);
      } else {
        setHasError(true);
        setErrorMessage('Unable to load forecast comparison data.');
      }
    } catch (err) {
      console.error('[ForecastChanges] Error loading data:', err);
      setHasError(true);
      setErrorMessage(err.message || 'Unable to load forecast comparison data.');
    } finally {
      setIsLoading(false);
    }
  }, [location, variable, leadTime, currentRun, compareRun]);

  useEffect(() => {
    fetchChangesData();
  }, [fetchChangesData]);

  const locObj = INDIAN_LOCATIONS.find((l) => l.id === location) || INDIAN_LOCATIONS[0];
  const locName = `${locObj.name}, ${locObj.state}`;

  const isSameRun = currentRun === compareRun;

  return (
    <div className="forecast-changes-page-container">
      {/* 1. Header */}
      <div className="overview-header changes-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Forecast Changes</h1>
          <p className="overview-subtitle">
            Compare the latest forecast with previous runs and understand what changed
          </p>
        </div>

        <div className="overview-header-right">
          <div className="changes-header-telemetry-pill">
            <span className="telemetry-lbl">Latest Run:</span>
            <span className="telemetry-time">14:00 IST</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-lbl">Compared Run:</span>
            <span className="telemetry-time">08:00 IST (-6h)</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-status-live">
              <span className="live-status-dot status-green"></span>
              Inter-Run Delta Active
            </span>
          </div>
        </div>
      </div>

      {/* 2. Run Selector & Filters */}
      <ForecastChangesFilters
        selectedLocation={location}
        onLocationChange={setLocation}
        selectedVariable={variable}
        onVariableChange={setVariable}
        selectedLeadTime={leadTime}
        onLeadTimeChange={setLeadTime}
        selectedCurrentRun={currentRun}
        onCurrentRunChange={setCurrentRun}
        selectedCompareRun={compareRun}
        onCompareRunChange={setCompareRun}
        isLoading={isLoading}
      />

      {/* 3. Inline Loading Indicator */}
      {isLoading && (
        <div className="forecast-inline-loader">
          <RefreshCw size={14} className="spinner-icon" />
          <span>Computing cross-run forecast differentials and model weight migrations...</span>
        </div>
      )}

      {/* 4. Error State */}
      {hasError ? (
        <div className="forecast-error-box">
          <AlertTriangle size={28} className="error-icon" />
          <div className="error-text-content">
            <h4>Unable to load forecast comparison data.</h4>
            <p>{errorMessage || 'The forecast run comparison service could not be reached.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchChangesData}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : isSameRun ? (
        /* Same Run Selected State */
        <div className="overview-placeholder-container">
          <div className="placeholder-box">
            <GitCompare size={36} color="#64748b" />
            <h3>Identical forecast runs selected for comparison.</h3>
            <p>Please select an earlier forecast run in the "Compare With" filter to inspect numerical differentials.</p>
            <button className="placeholder-back-btn" onClick={() => setCompareRun('run-0800')}>
              <span>Compare with Previous Run (-6h)</span>
            </button>
          </div>
        </div>
      ) : !changesData ? (
        /* Empty / No Data State */
        <div className="overview-placeholder-container">
          <div className="placeholder-box">
            <Database size={36} color="#64748b" />
            <h3>No previous forecast run is available for comparison.</h3>
            <p>Forecast-change analysis requires at least two initialized model runs.</p>
          </div>
        </div>
      ) : (
        /* 5. Core Operational Changes Content */
        <div className={`changes-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Section 4: Change Summary Cards */}
          <section className="changes-section">
            <ChangeSummaryCards
              activeVarChange={changesData.activeVarChange}
              runMeta={changesData.runMeta}
            />
          </section>

          {/* Section 5: Change Across Meteorological Variables */}
          <section className="changes-section">
            <ChangeByVariable
              variableChanges={changesData.variableChanges}
              onSelectVariable={setVariable}
              selectedVariable={variable}
            />
          </section>

          {/* Section 6: Forecast Evolution Across Horizons */}
          <section className="changes-section">
            <ChangeByHorizon
              horizonChanges={changesData.horizonChanges}
              unit={changesData.activeVarChange?.unit}
              currentLeadTime={leadTime}
              onSelectHorizon={setLeadTime}
            />
          </section>

          {/* Section 7: Source Model Forecast Adjustments */}
          <section className="changes-section">
            <ModelByModelChange
              modelChanges={changesData.modelChanges}
              unit={changesData.activeVarChange?.unit}
            />
          </section>

          {/* Section 8: SkyCast Blend Evolution & Weights Rebalancing */}
          <section className="changes-section">
            <SkyCastBlendChange
              weightChanges={changesData.weightChanges}
              activeVarChange={changesData.activeVarChange}
            />
          </section>

          {/* Sections 9, 10 & 12: Two-column grid (Uncertainty/Regime Shift & Significant Changes) */}
          <section className="changes-section two-column-grid">
            <UncertaintyAndRegimeChange
              uncertaintyChange={changesData.uncertaintyChange}
              regimeChange={changesData.regimeChange}
            />
            <SignificantChanges
              significantChanges={changesData.significantChanges}
            />
          </section>

          {/* Section 11: "What Changed?" Interpretation */}
          <section className="changes-section">
            <ChangeInterpretation
              interpretation={changesData.interpretation}
              locationName={locName}
            />
          </section>
        </div>
      )}
    </div>
  );
}
