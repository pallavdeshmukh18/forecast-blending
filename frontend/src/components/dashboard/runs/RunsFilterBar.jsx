import React from 'react';
import { Search, Filter, RefreshCw } from 'lucide-react';

export default function RunsFilterBar({
  searchQuery,
  onSearchChange,
  selectedStatus,
  onStatusChange,
  selectedHorizon,
  onHorizonChange,
  onResetFilters,
  totalRunsCount,
  filteredRunsCount,
}) {
  return (
    <div className="forecast-filters-panel runs-filter-bar">
      <div className="filter-group-row">
        {/* Search by Run ID */}
        <div className="filter-input-wrap">
          <label className="filter-control-label">Search Run</label>
          <div className="runs-search-box">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              placeholder="Filter by Run ID (e.g. 1400)..."
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              className="runs-search-input"
            />
          </div>
        </div>

        {/* Status Filter */}
        <div className="filter-input-wrap">
          <label className="filter-control-label">Run Status</label>
          <select
            value={selectedStatus}
            onChange={(e) => onStatusChange(e.target.value)}
            className="filter-select-input"
          >
            <option value="all">All Statuses</option>
            <option value="Completed">Completed</option>
            <option value="Partial">Partial / Warning</option>
            <option value="Running">Running</option>
            <option value="Failed">Failed</option>
          </select>
        </div>

        {/* Forecast Horizon Filter */}
        <div className="filter-input-wrap">
          <label className="filter-control-label">Forecast Horizon</label>
          <select
            value={selectedHorizon}
            onChange={(e) => onHorizonChange(e.target.value)}
            className="filter-select-input"
          >
            <option value="all">All Horizons</option>
            <option value="+72h">+72h (Full Extended)</option>
            <option value="+48h">+48h (Medium Range)</option>
            <option value="+24h">+24h (Short Range)</option>
          </select>
        </div>

        {/* Filter count & Reset */}
        <div className="filter-reset-wrap">
          <span className="runs-counter-tag">
            Showing <strong>{filteredRunsCount}</strong> of {totalRunsCount} runs
          </span>
          {(searchQuery || selectedStatus !== 'all' || selectedHorizon !== 'all') && (
            <button
              type="button"
              className="runs-reset-btn"
              onClick={onResetFilters}
              title="Reset search and filters"
            >
              Reset
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
