import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import ApproachSection from './components/ApproachSection';
import InteractiveMapSection, { CITIES_DATA } from './components/InteractiveMapSection';
import WhySkyCastSection from './components/WhySkyCastSection';
import FooterCta from './components/FooterCta';
import Footer from './components/Footer';
import LoginModal from './components/LoginModal';
import Dashboard from './components/dashboard/Dashboard';

export default function App() {
  const [currentView, setCurrentView] = useState(() => {
    if (typeof window !== 'undefined' && window.location.hash === '#dashboard') {
      return 'dashboard';
    }
    return 'landing';
  });
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [user, setUser] = useState({
    name: 'Dr. Anuj G.',
    role: 'Lead Meteorologist',
    email: 'meteorologist@skycast.ai'
  });
  const [currentCityKey, setCurrentCityKey] = useState('mumbai');

  // Listen for hash navigation
  useEffect(() => {
    const checkHash = () => {
      if (window.location.hash === '#dashboard') {
        setCurrentView('dashboard');
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
    setIsLoginModalOpen(false);
    setCurrentView('dashboard');
    window.location.hash = '#dashboard';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleLogout = () => {
    setCurrentView('landing');
    window.location.hash = '#home';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Handle location search from the top navbar
  const handleSearchLocation = (query) => {
    const q = query.toLowerCase().trim();
    const foundKey = Object.keys(CITIES_DATA).find(
      (k) =>
        k.includes(q) ||
        CITIES_DATA[k].city.toLowerCase().includes(q) ||
        CITIES_DATA[k].state.toLowerCase().includes(q)
    );

    if (foundKey) {
      setCurrentCityKey(foundKey);
    } else {
      alert(`Showing closest weather radar data for: "${query}". Defaulting to Mumbai blended model station.`);
      setCurrentCityKey('mumbai');
    }

    const forecastElement = document.getElementById('forecast');
    if (forecastElement) {
      forecastElement.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleExploreClick = (e) => {
    if (e) e.preventDefault();
    handleLogin();
  };

  const currentCityWeather = CITIES_DATA[currentCityKey] || CITIES_DATA.mumbai;

  // Render Operational Dashboard View
  if (currentView === 'dashboard') {
    return <Dashboard user={user} onLogout={handleLogout} />;
  }

  // Render Public Landing Page
  return (
    <div className="skycast-app">
      <Header
        onSearchLocation={handleSearchLocation}
        currentLocation={currentCityWeather.city}
        onLogin={() => handleLogin()}
        onOpenLogin={() => handleLogin()}
      />

      <main>
        <Hero
          weatherData={currentCityWeather}
          onExploreClick={handleExploreClick}
        />

        <ApproachSection />

        <InteractiveMapSection
          currentCityKey={currentCityKey}
          onSelectCity={setCurrentCityKey}
        />

        <WhySkyCastSection />

        <FooterCta onExploreClick={handleExploreClick} />
      </main>

      <Footer />

      {/* Login Modal fallback */}
      <LoginModal
        isOpen={isLoginModalOpen}
        onClose={() => setIsLoginModalOpen(false)}
        onLoginSuccess={handleLogin}
      />
    </div>
  );
}
