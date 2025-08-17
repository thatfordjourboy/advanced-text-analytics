import React, { useState, useEffect } from 'react';
import { 
  Brain, BarChart3, TrendingUp, Clock, 
  Loader2, RefreshCw, Download, 
  Eye, EyeOff, Radio, 
  Play, Pause, Globe, ChevronRight
} from 'lucide-react';
import { apiService } from '../services/api';

interface EmotionResult {
  text: string;
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
    disgust: number;
    neutral: number;
  };
  model_used: string;
  confidence: number;
  processing_time: number;
  timestamp: string;
  primary_emotion: string;
}

interface SampleHeadline {
  id: number;
  text: string;
  category: string;
  source: string;
  timestamp: string;
  emotion_hint: string;
}

const NewsHeadlines: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [selectedModel, setSelectedModel] = useState<string>('logistic_regression');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<EmotionResult[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentHeadline, setCurrentHeadline] = useState(0);

  // Sample headlines for quick start
  const sampleHeadlines: SampleHeadline[] = [
    {
      id: 1,
      text: "Breaking: Global markets surge to record highs as tech companies report unprecedented earnings growth",
      category: "Business",
      source: "Financial Times",
      timestamp: "2 hours ago",
      emotion_hint: "joy, surprise"
    },
    {
      id: 2,
      text: "Heartbreaking: Local community devastated by natural disaster, rescue efforts continue around the clock",
      category: "Local News",
      source: "City Herald",
      timestamp: "4 hours ago",
      emotion_hint: "sadness, fear"
    },
    {
      id: 3,
      text: "Outrage: Government officials face public backlash over controversial policy decision",
      category: "Politics",
      source: "National Post",
      timestamp: "6 hours ago",
      emotion_hint: "anger, disgust"
    }
  ];

  useEffect(() => {
    // Auto-rotate headlines
    const interval = setInterval(() => {
      if (isPlaying) {
        setCurrentHeadline(prev => (prev + 1) % sampleHeadlines.length);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isPlaying, sampleHeadlines.length]);

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    
    setIsAnalyzing(true);
    try {
      // Call the backend API using the apiService
      const response = await apiService.detectEmotion(inputText, selectedModel);
      
      if (!response.data) {
        throw new Error(response.error || 'Analysis failed');
      }

      const data = response.data;
      
      // Create a new result object
      const newResult: EmotionResult = {
        text: inputText,
        emotions: data.emotions,
        model_used: data.model_used || selectedModel,
        confidence: data.confidence || 0.85,
        processing_time: data.processing_time || 0.5,
        timestamp: new Date().toISOString(),
        primary_emotion: data.primary_emotion || 'neutral'
      };

      // Add to results
      setResults(prev => [newResult, ...prev]);
      
      // Clear input
      setInputText('');
      
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-white text-center mb-8">
          News Emotion Analysis
        </h1>
        <p className="text-xl text-slate-300 text-center mb-12">
          Advanced AI-powered emotion detection for news content
        </p>
        
        {/* Live News Ticker */}
        <div className="mb-8">
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-600/30 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-500 rounded-xl flex items-center justify-center">
                  <Radio className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-xl font-bold text-white">Live News Ticker</h2>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-2 rounded-lg bg-slate-700/50 text-slate-300 hover:text-white hover:bg-slate-600/50 transition-all"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <div className="flex items-center space-x-1 text-xs text-slate-400">
                  <Globe className="w-3 h-3" />
                  <span>Global Feed</span>
                </div>
              </div>
            </div>
            
            <div className="relative overflow-hidden h-16">
              <div className="flex items-center space-x-6 animate-text-scroll">
                {sampleHeadlines.map((headline, index) => (
                  <div
                    key={headline.id}
                    className={`flex items-center space-x-3 px-4 py-2 rounded-lg cursor-pointer transition-all ${
                      currentHeadline === index 
                        ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-400/30' 
                        : 'bg-slate-700/30 hover:bg-slate-700/50'
                    }`}
                    onClick={() => setInputText(headline.text)}
                  >
                    <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
                    <span className="text-white font-medium text-sm">{headline.category}</span>
                    <span className="text-slate-300 text-sm">{headline.source}</span>
                    <span className="text-slate-400 text-xs">{headline.timestamp}</span>
                    <ChevronRight className="w-4 h-4 text-slate-400" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Interface */}
        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {/* Input Panel */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">
                  Text Analysis
                </h2>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-300 hover:text-white transition-colors"
                  >
                    {showAdvanced ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    <span>{showAdvanced ? 'Hide' : 'Show'} Advanced</span>
                  </button>
                </div>
              </div>

              {/* Model Selection */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-white mb-3">
                  Model Selection
                </label>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    {
                      name: 'logistic_regression',
                      description: 'Fast and efficient for straightforward text analysis',
                      accuracy: 0.87,
                      training_samples: 15000,
                      status: 'default'
                    },
                    {
                      name: 'random_forest',
                      description: 'Robust and accurate for complex emotional content',
                      accuracy: 0.91,
                      training_samples: 15000,
                      status: 'default'
                    }
                  ].map((model) => (
                    <div
                      key={model.name}
                      className={`relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-300 ${
                        selectedModel === model.name
                          ? 'border-blue-400 bg-blue-500/20'
                          : 'border-slate-600 bg-slate-700/50 hover:border-slate-500'
                      }`}
                      onClick={() => setSelectedModel(model.name)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-white capitalize">
                          {model.name.replace('_', ' ')}
                        </span>
                        {model.status === 'default' && (
                          <span className="text-xs px-2 py-1 bg-emerald-500/20 text-emerald-300 rounded-full font-medium border border-emerald-400/30">
                            Default
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-300 mb-2">{model.description}</p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Accuracy: {(model.accuracy * 100).toFixed(0)}%</span>
                        <span className="text-slate-400">{model.training_samples.toLocaleString()} samples</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Text Input */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-white mb-3">
                  Enter Text for Analysis
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter your text here for emotion analysis..."
                  className="w-full p-4 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 resize-none h-32 focus:border-blue-400 focus:outline-none transition-all"
                  disabled={isAnalyzing}
                />
              </div>

              {/* Analysis Button */}
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !inputText.trim()}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white text-lg py-4 rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-3 animate-spin inline" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="w-5 h-5 mr-3 inline" />
                    Analyze Emotions
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Quick Stats Panel */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800/50 rounded-2xl p-6 h-fit border border-slate-600/30 backdrop-blur-xl">
              <h3 className="text-lg font-bold text-white mb-6">
                Analysis Statistics
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-500/10 to-indigo-500/10 rounded-xl border border-blue-400/20">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-indigo-500/20 rounded-xl flex items-center justify-center">
                      <BarChart3 className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300">Total Analyses</div>
                      <div className="text-2xl font-bold text-white">{results.length}</div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 rounded-xl border border-emerald-400/20">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-emerald-500/20 to-blue-500/20 rounded-xl flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300">Avg Confidence</div>
                      <div className="text-2xl font-bold text-white">
                        {results.length > 0 
                          ? (results.reduce((sum, r) => sum + r.confidence, 0) / results.length).toFixed(2)
                          : '0.00'
                        }
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl border border-purple-400/20">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center">
                      <Clock className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300">Avg Time</div>
                      <div className="text-2xl font-bold text-white">
                        {results.length > 0 
                          ? (results.reduce((sum, r) => sum + r.processing_time, 0) / results.length).toFixed(2)
                          : '0.00'
                        }s
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="mt-6 space-y-3">
                <button
                  onClick={() => alert('Export functionality coming soon!')}
                  disabled={results.length === 0}
                  className="w-full flex items-center justify-center px-4 py-2 bg-slate-700/50 text-white rounded-lg font-medium hover:bg-slate-600/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-slate-600"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export Results
                </button>
                <button
                  onClick={() => setResults([])}
                  disabled={results.length === 0}
                  className="w-full flex items-center justify-center px-4 py-2 bg-rose-500/20 text-rose-300 rounded-lg font-medium hover:bg-rose-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-rose-500/30"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Clear All
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Results Display */}
        {results.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-white text-center mb-6">
              Analysis Results
            </h2>
            
            {results.map((result, index) => (
              <div key={index} className="bg-slate-800/50 rounded-2xl p-6 border border-slate-600/30 backdrop-blur-xl">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center">
                      <Brain className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Text Analysis</h3>
                      <p className="text-sm text-slate-400">Model: {result.model_used.replace('_', ' ').toUpperCase()}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-slate-400">Processing Time</div>
                    <div className="text-lg font-bold text-blue-400">{result.processing_time.toFixed(3)}s</div>
                  </div>
                </div>
                
                {/* Original Text */}
                <div className="mb-6 p-4 bg-slate-700/50 rounded-xl border border-slate-600/30">
                  <p className="text-white italic">"{result.text}"</p>
                </div>
                
                {/* Primary Emotion */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-slate-300">Primary Emotion</span>
                    <span className="text-lg font-bold text-emerald-400 capitalize">{result.primary_emotion}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-3">
                    <div 
                      className="bg-emerald-500 h-3 rounded-full transition-all duration-500" 
                      style={{ width: `${(result.confidence * 100)}%` }}
                    ></div>
                  </div>
                  <div className="text-right text-sm text-slate-400 mt-1">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                
                {/* All Emotions Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(result.emotions).map(([emotion, confidence]) => (
                    <div key={emotion} className="text-center p-3 bg-slate-700/30 rounded-xl border border-slate-600/30">
                      <div className="text-sm font-medium text-slate-300 capitalize mb-2">{emotion}</div>
                      <div className="text-lg font-bold text-white mb-2">{(confidence * 100).toFixed(1)}%</div>
                      <div className="w-full bg-slate-600 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-500 ${
                            emotion === result.primary_emotion 
                              ? 'bg-emerald-500' 
                              : confidence > 0.5 
                                ? 'bg-blue-500' 
                                : 'bg-slate-500'
                          }`}
                          style={{ width: `${(confidence * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Model Details */}
                <div className="mt-6 pt-4 border-t border-slate-600/30">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-slate-400">Model Used:</span>
                      <div className="text-white font-medium capitalize">{result.model_used.replace('_', ' ')}</div>
                    </div>
                    <div>
                      <span className="text-slate-400">Processing Time:</span>
                      <div className="text-white font-medium">{result.processing_time.toFixed(3)}s</div>
                    </div>
                    <div>
                      <span className="text-slate-400">Timestamp:</span>
                      <div className="text-white font-medium">{new Date(result.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <div>
                      <span className="text-slate-400">Overall Confidence:</span>
                      <div className="text-white font-medium">{(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {/* Empty State */}
        {results.length === 0 && !isAnalyzing && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-br from-slate-600 to-slate-700 rounded-3xl flex items-center justify-center mx-auto mb-6">
              <Brain className="w-12 h-12 text-slate-300" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">
              Ready for Analysis
            </h3>
            <p className="text-slate-300 max-w-md mx-auto">
              Enter your text above or click on any headline from the live ticker to start analyzing emotions. 
              Our advanced ML models will provide comprehensive multi-label classification results.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default NewsHeadlines;
