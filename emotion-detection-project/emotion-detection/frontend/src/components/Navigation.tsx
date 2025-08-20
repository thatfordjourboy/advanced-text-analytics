import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Brain, Menu, X, BarChart3, Settings, Zap, FileText } from 'lucide-react';

const Navigation: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: Brain, description: 'Overview & Dashboard' },
    { path: '/detect-emotions', label: 'Emotion Detection', icon: Zap, description: 'Real-time Analysis' },
    { path: '/analytics', label: 'Analytics', icon: BarChart3, description: 'Data Insights' },
    { path: '/training', label: 'Model Training', icon: Settings, description: 'Train & Manage' },

    { path: '/docs', label: 'Documentation', icon: FileText, description: 'API & Guides' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-slate-900/90 border-b border-slate-700/50 sticky top-0 z-50 backdrop-blur-xl shadow-lg">
      <div className="container mx-auto px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center shadow-xl">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div className="hidden sm:block">
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Emotion Detection ML
              </span>
              <p className="text-xs text-slate-500 -mt-1">Powered by GloVe & ML</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const active = isActive(item.path)
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`group relative px-4 py-2 rounded-xl transition-all duration-300 ${
                    active 
                      ? 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 text-blue-300 border border-blue-400/30 shadow-sm' 
                      : 'text-slate-300 hover:text-blue-300 hover:bg-gradient-to-r hover:from-blue-500/10 hover:to-indigo-500/10 hover:shadow-sm'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className={`w-5 h-5 transition-all duration-300 ${
                      active ? 'text-blue-600 scale-110' : 'text-slate-500 group-hover:text-blue-600 group-hover:scale-110'
                    }`} />
                    <span className="font-semibold text-sm">{item.label}</span>
                  </div>
                  
                  {/* Enhanced Tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-3 px-3 py-2 bg-gradient-to-r from-slate-800 to-slate-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none whitespace-nowrap shadow-lg border border-slate-700">
                    {item.description}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800"></div>
                  </div>
                </Link>
              )
            })}
          </div>

          {/* Mobile menu button */}
          <div className="lg:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-3 rounded-xl text-slate-300 hover:bg-gradient-to-r hover:from-blue-500/10 hover:to-indigo-500/10 transition-all duration-300"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="lg:hidden py-6 border-t border-slate-700 bg-slate-900/95 backdrop-blur-md">
            <div className="grid grid-cols-2 gap-4">
              {navItems.map((item) => {
                const Icon = item.icon
                const active = isActive(item.path)
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`p-4 rounded-xl text-left transition-all duration-300 ${
                      active 
                        ? 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 text-blue-300 border border-blue-400/30 shadow-sm' 
                        : 'text-slate-300 hover:bg-gradient-to-r hover:from-blue-500/10 hover:to-indigo-500/10 hover:shadow-sm'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className={`w-6 h-6 ${
                        active ? 'text-blue-600' : 'text-slate-500'
                      }`} />
                      <div>
                        <div className="font-medium text-sm">{item.label}</div>
                        <div className="text-xs text-slate-500">{item.description}</div>
                      </div>
                    </div>
                  </Link>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

export default Navigation;
