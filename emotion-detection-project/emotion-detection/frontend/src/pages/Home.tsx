import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Brain, Zap, BarChart3, Settings, Sparkles, 
  Globe, Rocket, ArrowRight, Play, Pause, Volume2
} from 'lucide-react';

const Home: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentEmotion, setCurrentEmotion] = useState(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const emotions = [
    { name: 'Joy', color: 'from-yellow-400 to-orange-500', emoji: '😊', wave: 'rgba(255, 215, 0, 0.3)' },
    { name: 'Sadness', color: 'from-blue-400 to-indigo-500', emoji: '😢', wave: 'rgba(79, 172, 254, 0.3)' },
    { name: 'Anger', color: 'from-red-400 to-pink-500', emoji: '😠', wave: 'rgba(250, 112, 154, 0.3)' },
    { name: 'Fear', color: 'from-purple-400 to-indigo-500', emoji: '😨', wave: 'rgba(161, 140, 209, 0.3)' },
    { name: 'Surprise', color: 'from-orange-400 to-red-500', emoji: '😲', wave: 'rgba(255, 154, 158, 0.3)' },
    { name: 'Disgust', color: 'from-emerald-400 to-teal-500', emoji: '🤢', wave: 'rgba(255, 236, 210, 0.3)' }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentEmotion((prev) => (prev + 1) % emotions.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [emotions.length]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const createParticle = (x: number, y: number, color: string) => {
    const particle = document.createElement('div');
    particle.className = 'absolute w-2 h-2 rounded-full pointer-events-none';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    particle.style.backgroundColor = color;
    particle.style.animation = 'particle-float 2s ease-out forwards';
    document.body.appendChild(particle);
    
    setTimeout(() => {
      document.body.removeChild(particle);
    }, 2000);
  };

  const handleMouseClick = (e: React.MouseEvent) => {
    const colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a'];
    const randomColor = colors[Math.floor(Math.random() * colors.length)];
    createParticle(e.clientX, e.clientY, randomColor);
  };

  return (
    <div 
      className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden cursor-none"
      onClick={handleMouseClick}
    >
      {/* Interactive Background */}
      <div className="absolute inset-0">
        {/* Floating Geometric Shapes */}
        <div className="absolute top-20 left-20 w-32 h-32 border border-blue-400/20 rounded-full animate-spin-slow"></div>
        <div className="absolute top-40 right-40 w-24 h-24 bg-gradient-to-br from-purple-400/10 to-pink-400/10 rounded-full animate-pulse"></div>
        <div className="absolute bottom-40 left-1/3 w-20 h-20 border border-emerald-400/20 transform rotate-45 animate-bounce"></div>
        
        {/* Emotion Waves */}
        {emotions.map((emotion, index) => (
          <div
            key={emotion.name}
            className={`absolute inset-0 transition-opacity duration-1000 ${
              index === currentEmotion ? 'opacity-100' : 'opacity-0'
            }`}
          >
            <div 
              className="absolute inset-0"
              style={{
                background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, ${emotion.wave} 0%, transparent 50%)`
              }}
            ></div>
          </div>
        ))}
      </div>

      {/* Custom Cursor */}
      <div 
        className="fixed w-6 h-6 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full pointer-events-none z-50 transition-transform duration-100 ease-out"
        style={{
          left: mousePosition.x - 12,
          top: mousePosition.y - 12,
          transform: 'scale(1.2)'
        }}
      ></div>

      {/* Main Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        {/* Hero Section */}
        <section className="pt-20 pb-16 text-center">
          {/* Animated Logo */}
          <div className="flex justify-center mb-12">
            <div className="relative group">
              <div className="w-32 h-32 bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-600 rounded-full flex items-center justify-center shadow-2xl animate-pulse-glow group-hover:scale-110 transition-transform duration-500">
                <Brain className="w-16 h-16 text-white" />
              </div>
              
              {/* Orbiting Elements */}
              <div className="absolute inset-0 animate-spin-slow">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-emerald-400 rounded-full"></div>
                <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 w-4 h-4 bg-pink-400 rounded-full"></div>
                <div className="absolute left-0 top-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-yellow-400 rounded-full"></div>
                <div className="absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-blue-400 rounded-full"></div>
              </div>
              
              {/* Glow Effect */}
              <div className="absolute -inset-8 bg-gradient-to-br from-blue-400/30 to-purple-400/30 rounded-full blur-2xl animate-pulse-glow" style={{ animationDelay: '1s' }}></div>
            </div>
          </div>

          {/* Dynamic Title */}
          <div className="mb-8">
            <h1 className="text-6xl md:text-8xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white via-blue-100 to-purple-100 mb-4 leading-tight">
              Emotion
              <span className="block bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
                Sense
              </span>
            </h1>
            
            {/* Animated Subtitle */}
            <div className="h-8 overflow-hidden">
              <div className="animate-text-scroll">
                <p className="text-xl md:text-2xl text-slate-300 mb-2">Advanced emotion detection from news headlines</p>
                <p className="text-xl md:text-2xl text-slate-300 mb-2">Multi-label classification platform</p>
                <p className="text-xl md:text-2xl text-slate-300 mb-2">Built with help from GloVe</p>
              </div>
            </div>
          </div>

          {/* Interactive Controls */}
          <div className="flex items-center justify-center space-x-4 mb-12">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="w-12 h-12 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/20 transition-all duration-300"
            >
              {isPlaying ? <Pause className="w-5 h-5 text-white" /> : <Play className="w-5 h-5 text-white" />}
            </button>
            <div className="w-32 h-1 bg-white/20 rounded-full overflow-hidden">
              <div className="w-full h-full bg-gradient-to-r from-blue-400 to-purple-400 animate-pulse"></div>
            </div>
            <button className="w-12 h-12 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/20 transition-all duration-300">
              <Volume2 className="w-5 h-5 text-white" />
            </button>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center mb-20">
            <Link to="/headlines" className="group relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur-xl group-hover:blur-2xl transition-all duration-500"></div>
              <div className="relative px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold rounded-2xl text-lg transform group-hover:scale-105 transition-all duration-300">
                <Zap className="w-6 h-6 mr-3 inline-block group-hover:rotate-12 transition-transform" />
                Start Analysis
                <ArrowRight className="w-5 h-5 ml-2 inline-block group-hover:translate-x-1 transition-transform" />
              </div>
            </Link>
            
            <Link to="/analytics" className="group relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-600 to-blue-600 rounded-2xl blur-xl group-hover:blur-2xl transition-all duration-500"></div>
              <div className="relative px-8 py-4 bg-gradient-to-r from-emerald-600 to-blue-600 text-white font-bold rounded-2xl text-lg transform group-hover:scale-105 transition-all duration-300">
                <BarChart3 className="w-6 h-6 mr-3 inline-block group-hover:rotate-12 transition-transform" />
                View Analytics
              </div>
            </Link>
          </div>
        </section>

        {/* Interactive Features Grid */}
        <section className="py-20">
          <div className="grid md:grid-cols-3 gap-8">
            {emotions.map((emotion, index) => (
              <div
                key={emotion.name}
                className={`group relative overflow-hidden rounded-3xl p-8 transition-all duration-700 transform hover:scale-105 ${
                  index === currentEmotion ? 'bg-gradient-to-br from-white/10 to-white/5' : 'bg-white/5'
                }`}
                style={{
                  border: `1px solid ${index === currentEmotion ? 'rgba(255, 255, 255, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`
                }}
              >
                {/* Background Pattern */}
                <div className="absolute inset-0 opacity-20">
                  <div className="absolute inset-0" style={{
                    backgroundImage: `radial-gradient(circle at 20% 80%, ${emotion.wave} 0%, transparent 50%)`
                  }}></div>
                </div>
                
                <div className="relative z-10 text-center">
                  <div className={`w-20 h-20 bg-gradient-to-br ${emotion.color} rounded-3xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-500`}>
                    <span className="text-4xl">{emotion.emoji}</span>
                  </div>
                  
                  <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-blue-400 group-hover:to-purple-400 transition-all duration-500">
                    {emotion.name}
                  </h3>
                  
                  <p className="text-slate-300 leading-relaxed">
                    Advanced detection algorithms for {emotion.name.toLowerCase()} classification with real-time analysis capabilities.
                  </p>
                  
                  {/* Hover Effect */}
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-400/5 to-purple-400/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-3xl"></div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Floating Action Cards */}
        <section className="py-20 relative">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: <Brain className="w-8 h-8" />, title: 'ML Models', desc: 'Advanced algorithms', color: 'from-blue-500/20 to-indigo-500/20' },
              { icon: <Zap className="w-8 h-8" />, title: 'Real-time', desc: 'Instant results', color: 'from-emerald-500/20 to-blue-500/20' },
              { icon: <BarChart3 className="w-8 h-8" />, title: 'Analytics', desc: 'Deep insights', color: 'from-purple-500/20 to-pink-500/20' },
              { icon: <Settings className="w-8 h-8" />, title: 'Custom', desc: 'Tailored solutions', color: 'from-orange-500/20 to-red-500/20' }
            ].map((feature, index) => (
              <div
                key={feature.title}
                className="group relative overflow-hidden"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-white/2 rounded-2xl transform group-hover:scale-105 transition-all duration-500"></div>
                <div className="relative p-6 text-center">
                  <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-500`}>
                    <div className="text-blue-400">{feature.icon}</div>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">{feature.title}</h3>
                  <p className="text-sm text-slate-400">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="py-16 text-center">
          <div className="flex items-center justify-center mb-8">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mr-4">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              EmotionSense
            </span>
          </div>
          
          <p className="text-slate-400 mb-6">
            Pushing the boundaries of emotion detection with cutting-edge ML models
          </p>
          
          <div className="flex items-center justify-center space-x-6 text-sm text-slate-500">
            <div className="flex items-center">
              <Globe className="w-4 h-4 mr-2" />
              <span>GloVe Embeddings</span>
            </div>
            <div className="flex items-center">
              <Rocket className="w-4 h-4 mr-2" />
              <span>Logistic Regression</span>
            </div>
            <div className="flex items-center">
              <Sparkles className="w-4 h-4 mr-2" />
              <span>Random Forest</span>
            </div>
          </div>
        </footer>
      </div>


    </div>
  );
};

export default Home;
