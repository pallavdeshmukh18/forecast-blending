import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App Crash ErrorBoundary caught:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '40px',
          maxWidth: '800px',
          margin: '40px auto',
          background: '#0d131f',
          color: '#f1f5f9',
          fontFamily: 'system-ui, sans-serif',
          borderRadius: '12px',
          border: '1px solid #334155',
          boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
        }}>
          <h2 style={{ color: '#ef4444', marginTop: 0 }}>Something went wrong while rendering</h2>
          <p style={{ color: '#94a3b8' }}>{this.state.error?.message || 'Unknown error'}</p>
          <pre style={{
            background: '#1e293b',
            padding: '16px',
            borderRadius: '8px',
            overflowX: 'auto',
            fontSize: '12px',
            color: '#fca5a5'
          }}>
            {this.state.error?.stack}
          </pre>
          <button
            onClick={() => {
              window.location.hash = '#home';
              window.location.reload();
            }}
            style={{
              marginTop: '16px',
              padding: '10px 20px',
              background: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            Reset to Landing Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
