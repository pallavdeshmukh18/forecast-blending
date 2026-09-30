import React from 'react';
import { History, CheckCircle2, Clock } from 'lucide-react';

export default function RecentEventsLog({ events = [] }) {
  if (!events || events.length === 0) return null;

  return (
    <div className="forecast-explorer-card events-log-card">
      <div className="card-top-header">
        <div>
          <h3 className="card-heading">Recent System Events</h3>
          <p className="card-subheading">Operational execution log of model ingestion, blending, and publication runs</p>
        </div>
      </div>

      <div className="forecast-table-wrapper">
        <table className="forecast-table events-operational-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Originating Service</th>
              <th>System Event Details</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev) => (
              <tr key={ev.id}>
                <td>
                  <span className="font-mono text-secondary flex-align-center">
                    <Clock size={12} className="mr-1 inline text-slate-400" />
                    {ev.time}
                  </span>
                </td>
                <td>
                  <span className="font-semibold text-white">{ev.service}</span>
                </td>
                <td>
                  <span className="event-desc-text">{ev.event}</span>
                </td>
                <td>
                  <span className="status-badge-compact badge-completed">
                    <span className="status-indicator-dot dot-green"></span>
                    {ev.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
