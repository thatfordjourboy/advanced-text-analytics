import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import ErrorBoundary from './components/ErrorBoundary';
import Home from './pages/Home';
import Analytics from './pages/Analytics';
import ModelTraining from './pages/ModelTraining';

import NewsHeadlines from './pages/NewsHeadlines';
import ResearchTools from './pages/ResearchTools';
import Documentation from './pages/Documentation';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
          <Navigation />
          <main className="pt-16">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/training" element={<ModelTraining />} />

              <Route path="/detect-emotions" element={<NewsHeadlines />} />
              <Route path="/research" element={<ResearchTools />} />
              <Route path="/docs" element={<Documentation />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
