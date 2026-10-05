import React, { useState, useEffect } from 'react';
import BharatLanding from './components/BharatLanding';
import Dashboard from './components/dashboard/Dashboard';

export default function App() {
  const [currentView, setCurrentView] = useState(() => {
    if (typeof window !== 'undefined' && window.location.hash.startsWith('#dashboard')) {
      return 'dashboard';
    }
    return 'landing';
  });
  const [user, setUser] = useState({
    name: 'Dr. Anuj G.',
    role: 'Lead Meteorologist',
    email: 'meteorologist@skycast.ai'
  });

  // Listen for hash navigation
  useEffect(() => {
    const checkHash = () => {
      if (window.location.hash.startsWith('#dashboard')) {
        setCurrentView('dashboard');
      } else {
        setCurrentView('landing');
      }
    };
    checkHash();
    window.addEventListener('hashchange', checkHash);
    return () => window.removeEventListener('hashchange', checkHash);
  }, []);

  // Direct login - immediately lands inside the dashboard!
  const handleLogin = (customUser) => {
    if (customUser && customUser.name) {
      setUser(customUser);
    } else {
      setUser({
        name: 'Dr. Anuj G.',
        role: 'Lead Meteorologist',
        email: 'meteorologist@skycast.ai'
      });
    }
    setCurrentView('dashboard');
    window.location.hash = '#dashboard';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleLogout = () => {
    setCurrentView('landing');
    window.location.hash = '#home';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Render Operational Dashboard View
  if (currentView === 'dashboard') {
    return <Dashboard user={user} onLogout={handleLogout} />;
  }

  return <BharatLanding onGetStarted={() => handleLogin()} />;
}
