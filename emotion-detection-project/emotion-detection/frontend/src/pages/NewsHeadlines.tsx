import React, { useState, useEffect } from 'react';
import { 
  Brain, Loader2, TrendingUp, BarChart3, Clock, Download, RefreshCw, Zap,
  Newspaper, Globe, Activity, AlertCircle, CheckCircle, Settings, Play, Eye,
  Radio, Pause, ChevronRight, EyeOff, List, X
} from 'lucide-react';
import { apiService } from '../services/api';

interface EmotionResult {
  text: string;
  emotions: Record<string, number>;
  model_used: string;
  confidence: number;
  processing_time: number;
  timestamp: string;
  primary_emotion: string;
}

interface MultilineEmotionResult {
  text: string;
  overall_emotions: Record<string, number>;
  overall_primary_emotion: string;
  overall_confidence: number;
  sentence_analyses: Array<{
    paragraph_index?: number;
    sentence_index?: number;
    text: string;
    emotions: Record<string, number>;
    primary_emotion: string;
    confidence: number;
    word_count?: number;
    error?: string;
  }>;
  total_paragraphs?: number;
  total_sentences?: number;
  processing_time: number;
  model_used: string;
  analysis_type: string;
  timestamp: string;
}

interface SampleHeadline {
  id: string;
  title: string;
  description: string;
  category: string;
  source: string;
  timestamp: string;
  url: string;
  image: string;
  sentiment: string;
}

const NewsHeadlines: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [selectedModel, setSelectedModel] = useState<string>('logistic_regression');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<(EmotionResult | MultilineEmotionResult)[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentHeadlineIndex, setCurrentHeadlineIndex] = useState(0);
  const [refreshMode, setRefreshMode] = useState<'cache' | 'force'>('cache');
  const [lastRefreshClick, setLastRefreshClick] = useState<number>(0);
  const [isMultilineAnalysis, setIsMultilineAnalysis] = useState(false);
  const [liveNews, setLiveNews] = useState<SampleHeadline[]>([]);
  const [isLoadingNews, setIsLoadingNews] = useState(true);
  const [lastNewsUpdate, setLastNewsUpdate] = useState<Date | null>(null);
  const [newsStatus, setNewsStatus] = useState<string>('cached');
  const [nextRefresh, setNextRefresh] = useState<string>('4 hours');
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportFormats, setExportFormats] = useState<any[]>([]);
  const [isExporting, setIsExporting] = useState(false);
  const [selectedExportFormat, setSelectedExportFormat] = useState('json');

  // Type guards for result types
  const isMultilineResult = (result: EmotionResult | MultilineEmotionResult): result is MultilineEmotionResult => {
    return 'analysis_type' in result && (result.analysis_type === 'multiline' || result.analysis_type === 'multiline_paragraphs');
  };

  const isSingleLineResult = (result: EmotionResult | MultilineEmotionResult): result is EmotionResult => {
    return !isMultilineResult(result);
  };

  // Fetch live news function
  const fetchLiveNews = async () => {
    try {
      setIsLoadingNews(true);
      const response = await apiService.getLiveNews();
      
      if (response.data && response.data.articles) {
        setLiveNews(response.data.articles);
        setLastNewsUpdate(new Date());
        setNewsStatus(response.data.status || 'unknown');
        setNextRefresh(response.data.next_refresh || '4 hours');
      } else {
        // Fallback to sample headlines if API fails
        setLiveNews(sampleHeadlines);
        setLastNewsUpdate(new Date());
        setNewsStatus('fallback');
        setNextRefresh('1 hour');
      }
    } catch (error) {
      console.error('Failed to fetch live news:', error);
      // Fallback to sample headlines
      setLiveNews(sampleHeadlines);
      setLastNewsUpdate(new Date());
      setNewsStatus('error');
      setNextRefresh('1 hour');
    } finally {
      setIsLoadingNews(false);
    }
  };

  // Smart refresh function that combines cache and force refresh
  const handleSmartRefresh = async () => {
    const now = Date.now();
    const timeSinceLastClick = now - lastRefreshClick;
    
    // If clicked within 3 seconds, force refresh; otherwise check cache
    if (timeSinceLastClick < 3000 && refreshMode === 'cache') {
      setRefreshMode('force');
      await forceRefreshNews();
    } else {
      setRefreshMode('cache');
      await fetchLiveNews();
    }
    
    setLastRefreshClick(now);
    
    // Reset mode after 5 seconds
    setTimeout(() => setRefreshMode('cache'), 5000);
  };

  // Force refresh news function (bypasses cache)
  const forceRefreshNews = async () => {
    try {
      setIsLoadingNews(true);
      const response = await apiService.forceRefreshNews();
      
      if (response.data && response.data.articles) {
        setLiveNews(response.data.articles);
        setLastNewsUpdate(new Date());
        setNewsStatus(response.data.status || 'refreshed');
        setNextRefresh(response.data.next_refresh || '4 hours');
      }
    } catch (error) {
      console.error('Failed to force refresh news:', error);
      alert('Failed to refresh news. Using cached data instead.');
    } finally {
      setIsLoadingNews(false);
    }
  };

  // Load available export formats
  const loadExportFormats = async () => {
    try {
      const response = await apiService.getExportFormats();
      if (response.data && response.data.formats) {
        setExportFormats(response.data.formats);
      }
    } catch (error) {
      console.error('Failed to load export formats:', error);
    }
  };

  // Export results function
  const exportResults = async () => {
    if (results.length === 0) {
      alert('No results to export. Please analyze some text first.');
      return;
    }

    try {
      setIsExporting(true);
      
      // Prepare results for export (add timestamp if not present)
      const exportData = results.map(result => ({
        ...result,
        timestamp: result.timestamp || new Date().toISOString()
      }));

      const blob = await apiService.exportResults(exportData, selectedExportFormat);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `emotion_analysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.${selectedExportFormat}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setShowExportModal(false);
      alert('Export completed successfully!');
    } catch (error) {
      console.error('Export failed:', error);
      alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsExporting(false);
    }
  };

  // Sample headlines for quick start - updated with correct emotion names
  const sampleHeadlines: SampleHeadline[] = [
    {
      id: "1",
      title: "Breaking: Global markets surge to record highs as tech companies report unprecedented earnings growth",
      description: "Global markets surged to record highs today, driven by strong earnings from major tech companies. The Dow Jones Industrial Average gained 2.5%, while the S&P 500 rose 2.1%. Tech giants like Apple, Microsoft, and Google all reported impressive quarterly results, contributing to the overall market optimism.",
      category: "Business",
      source: "Financial Times",
      timestamp: "2 hours ago",
      url: "https://www.ft.com/content/...",
      image: "https://via.placeholder.com/150",
      sentiment: "positive"
    },
    {
      id: "2",
      title: "Heartbreaking: Local community devastated by natural disaster, rescue efforts continue around the clock",
      description: "A devastating natural disaster has left the local community devastated. Rescue efforts are underway around the clock, with multiple agencies working together to provide aid and support to those affected. The situation remains critical, and the community is rallying to help those in need.",
      category: "Local News",
      source: "City Herald",
      timestamp: "4 hours ago",
      url: "https://www.cityherald.com/content/...",
      image: "https://via.placeholder.com/150",
      sentiment: "negative"
    },
    {
      id: "3",
      title: "Outrage: Government officials face public backlash over controversial policy decision",
      description: "A controversial policy decision by government officials has sparked outrage among the public. Citizens are demanding transparency and accountability from the authorities, as the decision has implications for the entire nation. The situation is tense, and the government is facing increasing pressure to reconsider its stance.",
      category: "Politics",
      source: "National Post",
      timestamp: "6 hours ago",
      url: "https://www.nationalpost.com/content/...",
      image: "https://via.placeholder.com/150",
      sentiment: "anger"
    },
    {
      id: "4",
      title: "The city was alive with excitement as the festival began. Children laughed and played in the streets. Music filled the air with joy and celebration. Everyone felt a sense of community and happiness.",
      description: "The city was alive with excitement as the festival began. Children laughed and played in the streets. Music filled the air with joy and celebration. Everyone felt a sense of community and happiness.",
      category: "Community",
      source: "Local Gazette",
      timestamp: "8 hours ago",
      url: "https://www.localgazette.com/content/...",
      image: "https://via.placeholder.com/150",
      sentiment: "happiness"
    }
  ];

  useEffect(() => {
    // Auto-rotate headlines
    const interval = setInterval(() => {
      if (isPlaying && liveNews.length > 0) {
        setCurrentHeadlineIndex(prev => (prev + 1) % liveNews.length);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isPlaying, liveNews.length]);

  useEffect(() => {
    fetchLiveNews();
    // Refresh news every 4 hours instead of every 30 seconds to avoid API rate limits
    const newsInterval = setInterval(fetchLiveNews, 4 * 60 * 60 * 1000); // 4 hours
    return () => clearInterval(newsInterval);
  }, []);

  useEffect(() => {
    loadExportFormats();
  }, []);

  // Handle analyze function
  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      alert('Please enter some text to analyze.');
      return;
    }

    try {
      setIsAnalyzing(true);
      
      let result;
      console.log('Sending request:', { text: inputText, model: selectedModel, multiline: isMultilineAnalysis });
      if (isMultilineAnalysis) {
        result = await apiService.detectEmotionMultiline(inputText, selectedModel);
      } else {
        result = await apiService.detectEmotion(inputText, selectedModel);
      }
      console.log('Received result:', result);
      
      if (result && result.data) {
        const newResult = {
          ...result.data,
          timestamp: new Date().toISOString(),
          id: Date.now()
        };
        
        setResults(prev => [newResult, ...prev]);
        setInputText(''); // Clear input after successful analysis
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      
      // Better error handling to prevent [object Object] messages
      let errorMessage = 'Analysis failed';
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      } else if (error && typeof error === 'object' && 'message' in error) {
        errorMessage = String(error.message);
      } else {
        errorMessage = 'Unknown error occurred during analysis';
      }
      
      alert(errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Helper function to get color based on emotion
  const getEmotionColor = (emotion: string) => {
    switch (emotion) {
      case 'anger':
        return 'hsl(var(--red))';
      case 'disgust':
        return 'hsl(var(--orange))';
      case 'fear':
        return 'hsl(var(--purple))';
      case 'happiness':
        return 'hsl(var(--yellow))';
      case 'no emotion':
        return 'hsl(var(--gray))';
      case 'sadness':
        return 'hsl(var(--blue))';
      case 'surprise':
        return 'hsl(var(--pink))';
      default:
        return 'hsl(var(--slate))';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-white text-center mb-8">
          News Emotion Analysis
        </h1>
        <p className="text-xl text-slate-300 text-center mb-12">
          Advanced AI-powered 7-class emotion detection for news content
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
                {isLoadingNews && (
                  <div className="flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                    <span className="text-sm text-blue-400">Loading...</span>
                  </div>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex flex-col items-center">
                  <button
                    onClick={handleSmartRefresh}
                    disabled={isLoadingNews}
                    className={`p-2 rounded-lg transition-all disabled:opacity-50 border ${
                      refreshMode === 'force' 
                        ? 'bg-red-600/20 text-red-400 hover:text-white hover:bg-red-600/30 border-red-500/30' 
                        : 'bg-blue-600/20 text-blue-400 hover:text-white hover:bg-blue-600/30 border-blue-500/30'
                    }`}
                    title={refreshMode === 'force' ? 'Force Refresh (API call)' : 'Smart Refresh (Cache first)'}
                  >
                    <RefreshCw className={`w-4 h-4 ${isLoadingNews ? 'animate-spin' : ''}`} />
                  </button>
                  <div className="text-xs text-slate-400 mt-1">
                    {refreshMode === 'force' ? 'Force' : 'Smart'}
                  </div>
                </div>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-2 rounded-lg bg-slate-700/50 text-slate-300 hover:text-white hover:bg-slate-600/50 transition-all"
                  title={isPlaying ? 'Pause ticker' : 'Play ticker'}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <div className="flex items-center space-x-1 text-xs text-slate-400">
                  <Globe className="w-3 h-3" />
                  <span>Live Feed</span>
                </div>
              </div>
            </div>
            
            {/* Ticker Position Indicator */}
            <div className="flex justify-center mt-2 space-x-1">
              {liveNews.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full transition-all ${
                    index === currentHeadlineIndex
                      ? 'bg-blue-400 scale-125'
                      : 'bg-slate-600'
                  }`}
                />
              ))}
            </div>
            
            {/* Smart Refresh Instructions */}
            <div className="text-center mt-2">
              <p className="text-xs text-slate-400">
                💡 <strong>Smart Refresh:</strong> Click once for cache check, click again within 3 seconds for API refresh
              </p>
            </div>
            
            {/* Cache Status and Next Refresh */}
            <div className="flex items-center justify-between mb-3 text-xs">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <span className="text-slate-400">Status:</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    newsStatus === 'cached' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                    newsStatus === 'fresh' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
                    newsStatus === 'refreshed' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' :
                    newsStatus === 'fallback' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                    'bg-red-500/20 text-red-400 border border-red-500/30'
                  }`}>
                    {newsStatus.charAt(0).toUpperCase() + newsStatus.slice(1)}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-slate-400">Next refresh:</span>
                  <span className="text-slate-300 font-medium">{nextRefresh}</span>
                </div>
              </div>
              
              {lastNewsUpdate && (
                <div className="text-slate-400">
                  Last updated: {lastNewsUpdate.toLocaleTimeString()}
                </div>
              )}
            </div>
            
            <div className="relative overflow-hidden h-16">
              <div 
                className="flex transition-transform duration-1000 ease-in-out"
                style={{ 
                  transform: `translateX(-${currentHeadlineIndex * 100}%)`,
                  width: `${liveNews.length * 100}%`
                }}
              >
                {liveNews.map((headline, index) => (
                  <div 
                    key={headline.id}
                    className="flex items-center space-x-3 px-4 py-2 rounded-lg cursor-pointer transition-all w-full flex-shrink-0"
                    onClick={() => {
                      if (isMultilineAnalysis) {
                        // For multi-line analysis, combine title and description to create paragraphs
                        const content = headline.description 
                          ? `${headline.title}\n\n${headline.description}`
                          : headline.title;
                        setInputText(content);
                      } else {
                        // For single-line analysis, just use the title
                        setInputText(headline.title);
                      }
                    }}
                  >
                    <div className="flex-shrink-0">
                      <div className={`w-2 h-2 rounded-full ${
                        headline.sentiment === 'positive' ? 'bg-green-400' :
                        headline.sentiment === 'negative' ? 'bg-red-400' :
                        'bg-slate-400'
                      }`}></div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white truncate">
                        {headline.title}
                      </div>
                      {isMultilineAnalysis && headline.description && (
                        <div className="text-xs text-slate-400 truncate mt-1">
                          {headline.description}
                        </div>
                      )}
                    </div>
                    <div className="flex-shrink-0 text-xs text-slate-500">
                      {headline.category}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {liveNews.length === 0 && !isLoadingNews && (
              <div className="text-center py-4 text-slate-400">
                No news available. Click refresh to try again.
              </div>
            )}
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
                
                {/* Balanced Metrics Disclaimer */}
                <div className="mb-4 p-3 bg-amber-500/10 border border-amber-400/30 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <div className="w-5 h-5 bg-amber-500/20 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-amber-400 text-xs">ℹ</span>
                    </div>
                    <div className="text-xs text-amber-200">
                      <p className="font-medium mb-1">Balanced Metrics for Imbalanced Data</p>
                      <p className="text-amber-300/80">
                        Due to severe class imbalance (491:1 ratio), we prioritize F1-score (macro) over accuracy. 
                        This ensures fair evaluation across all 7 emotion classes.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  {[
                    {
                      name: 'logistic_regression',
                      description: 'Fast and efficient for straightforward text analysis',
                      accuracy: 0.88,
                      f1_macro: 0.82,
                      f1_weighted: 0.87,
                      training_samples: 87170,
                      status: 'default'
                    },
                    {
                      name: 'random_forest',
                      description: 'Robust and accurate for complex emotional content',
                      accuracy: 0.82,
                      f1_macro: 0.78,
                      f1_weighted: 0.84,
                      training_samples: 87170,
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
                      <div className="space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">F1 (Macro):</span>
                          <span className="text-emerald-400 font-medium">{(model.f1_macro * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">F1 (Weighted):</span>
                          <span className="text-blue-400 font-medium">{(model.f1_weighted * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">Accuracy:</span>
                          <span className="text-slate-400">{(model.accuracy * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">Samples:</span>
                          <span className="text-slate-400">{model.training_samples.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Text Input */}
              <div className="mb-6">
                {/* Multi-line Analysis Toggle */}
                <div className="mb-4 flex items-center justify-between">
                  <label className="block text-sm font-semibold text-white">
                    Analysis Mode
                  </label>
                  <div className="flex items-center space-x-3">
                    <span className={`text-sm transition-colors ${!isMultilineAnalysis ? 'text-emerald-400' : 'text-slate-400'}`}>
                      Single Line
                    </span>
                    <button
                      onClick={() => setIsMultilineAnalysis(!isMultilineAnalysis)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                        isMultilineAnalysis ? 'bg-blue-600' : 'bg-slate-600'
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          isMultilineAnalysis ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                    <span className={`text-sm transition-colors ${isMultilineAnalysis ? 'text-blue-400' : 'text-slate-400'}`}>
                      Multi-Line
                    </span>
                  </div>
                </div>
                
                <label className="block text-sm font-semibold text-white mb-3">
                  {isMultilineAnalysis ? 'Enter Text for Multi-Line Analysis' : 'Enter Text for Analysis'}
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder={isMultilineAnalysis 
                    ? "Enter multiple sentences or paragraphs for detailed emotion analysis..." 
                    : "Enter your text here for emotion analysis..."
                  }
                  className="w-full p-4 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 resize-none h-32 focus:border-blue-400 focus:outline-none transition-all"
                  disabled={isAnalyzing}
                />
                {isMultilineAnalysis && (
                  <div className="mt-2 text-xs text-slate-400">
                    💡 Multi-line analysis will break down your text into paragraphs and analyze each one individually, 
                    providing both paragraph-level and overall emotion insights.
                  </div>
                )}
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
                    {isMultilineAnalysis ? 'Analyze Multi-Line Text' : 'Analyze Emotions'}
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
                          ? (results.reduce((sum, r) => {
                              if (isMultilineResult(r)) {
                                return sum + r.overall_confidence;
                              } else {
                                return sum + r.confidence;
                              }
                            }, 0) / results.length).toFixed(2)
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

        {/* Results Section */}
        {results.length > 0 && (
          <div className="space-y-6">
            {/* Export Button */}
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">
                Analysis Results ({results.length})
              </h3>
              <button
                onClick={() => setShowExportModal(true)}
                disabled={results.length === 0}
                className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>Export Results</span>
              </button>
            </div>
            
            {results.map((result, index) => {
              // Handle different result types
              if (isMultilineResult(result)) {
                return (
                  <div key={index} className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
                    {/* Header Section */}
                    <div className="flex items-start justify-between mb-6">
                      <div className="flex items-center space-x-4">
                        <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center">
                          <Brain className="w-8 h-8 text-white" />
                    </div>
                    <div>
                          <h3 className="text-2xl font-bold text-white">Multi-Line Analysis</h3>
                          <p className="text-lg text-slate-400">Model: {result.model_used.replace('_', ' ').toUpperCase()}</p>
                          <div className="flex items-center space-x-2 mt-2">
                            <Clock className="w-4 h-4 text-slate-400" />
                            <span className="text-sm text-slate-400">
                              {new Date(result.timestamp).toLocaleString()}
                            </span>
                          </div>
                    </div>
                  </div>
                  <div className="text-right">
                        <div className="text-sm text-slate-400 mb-1">Processing Time</div>
                        <div className="text-2xl font-bold text-blue-400">{result.processing_time.toFixed(3)}s</div>
                  </div>
                </div>
                
                {/* Original Text */}
                    <div className="mb-8 p-6 bg-slate-700/50 rounded-xl border border-slate-600/30">
                      <div className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                          <span className="text-xs text-blue-400">📄</span>
                        </div>
                        <div className="text-white text-lg leading-relaxed">
                          <p className="whitespace-pre-wrap">"{result.text}"</p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Overall Analysis Summary */}
                    <div className="mb-8 p-6 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 rounded-xl border border-emerald-400/30">
                      <h4 className="text-xl font-semibold text-white mb-4 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-emerald-400" />
                        Overall Analysis Summary
                      </h4>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-emerald-400">
                            {result.overall_primary_emotion.replace('_', ' ').toUpperCase()}
                          </div>
                          <div className="text-sm text-slate-400">Overall Primary Emotion</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-400">
                            {(result.overall_confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-slate-400">Overall Confidence</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-400">
                            {result.total_paragraphs || result.sentence_analyses?.length || 0}
                          </div>
                          <div className="text-sm text-slate-400">Paragraphs Analyzed</div>
                        </div>
                      </div>
                </div>
                
                    {/* Paragraph-by-Paragraph Analysis */}
                    <div className="space-y-4">
                      <h4 className="text-xl font-semibold text-white mb-4 flex items-center">
                        <List className="w-5 h-5 mr-2 text-blue-400" />
                        Paragraph-by-Paragraph Analysis
                      </h4>
                      {result.sentence_analyses?.map((paragraph, pIndex) => (
                        <div key={pIndex} className="p-4 bg-slate-700/30 rounded-lg border border-slate-600/30">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex-1">
                              <div className="text-sm text-slate-400 mb-1">Paragraph #{paragraph.paragraph_index || pIndex + 1}</div>
                              <p className="text-white leading-relaxed">"{paragraph.text}"</p>
                              {paragraph.word_count && (
                                <div className="text-xs text-slate-500 mt-1">
                                  {paragraph.word_count} words
                                </div>
                              )}
                            </div>
                            <div className="text-right ml-4">
                              <div className="text-lg font-bold text-emerald-400">
                                {paragraph.primary_emotion?.replace('_', ' ') || 'Unknown'}
                  </div>
                              <div className="text-sm text-slate-400">
                                {(paragraph.confidence * 100).toFixed(1)}% confidence
                  </div>
                  </div>
                </div>
                
                          {/* Emotion breakdown for this sentence */}
                          <div className="grid grid-cols-7 gap-2">
                            {paragraph.emotions && typeof paragraph.emotions === 'object' && Object.entries(paragraph.emotions).map(([emotion, confidence]) => (
                              <div key={emotion} className="text-center">
                                <div className="text-xs text-slate-400 mb-1">{emotion}</div>
                      <div className="w-full bg-slate-600 rounded-full h-2">
                        <div 
                                    className="h-2 rounded-full bg-blue-500 transition-all duration-300"
                          style={{ width: `${(confidence * 100)}%` }}
                        ></div>
                                </div>
                                <div className="text-xs text-white mt-1">
                                  {(confidence * 100).toFixed(0)}%
                                </div>
                              </div>
                            ))}
                            {(!paragraph.emotions || typeof paragraph.emotions !== 'object' || Object.keys(paragraph.emotions).length === 0) && (
                              <div className="col-span-7 text-center text-slate-400 text-xs py-2">
                                No emotion data available for this paragraph
                              </div>
                            )}
                      </div>
                    </div>
                  ))}
                </div>
                  </div>
                );
              } else {
                // Single-line result display (existing code)
                // Sort emotions by confidence (highest first) and filter out very low confidence emotions
                const sortedEmotions = result.emotions && typeof result.emotions === 'object' 
                  ? Object.entries(result.emotions)
                      .filter(([_, confidence]) => confidence > 0.01) // Only show emotions with >1% confidence
                      .sort(([_, a], [__, b]) => b - a)
                  : [];
                
                // Get top 3 emotions for primary display
                const topEmotions = sortedEmotions.slice(0, 3);
                
                // Define emotion colors for single-line results
                const emotionColors = {
                  anger: 'from-red-500 to-red-600',
                  disgust: 'from-orange-500 to-orange-600',
                  fear: 'from-purple-500 to-purple-600',
                  happiness: 'from-yellow-500 to-yellow-600',
                  'no emotion': 'from-gray-500 to-gray-600',
                  sadness: 'from-blue-500 to-blue-600',
                  surprise: 'from-pink-500 to-pink-600'
                };
                
                return (
                  <div key={index} className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
                    {/* Header Section */}
                    <div className="flex items-start justify-between mb-6">
                      <div className="flex items-center space-x-4">
                        <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center">
                          <Brain className="w-8 h-8 text-white" />
                        </div>
                    <div>
                          <h3 className="text-2xl font-bold text-white">Text Analysis</h3>
                          <p className="text-lg text-slate-400">Model: {result.model_used.replace('_', ' ').toUpperCase()}</p>
                          <div className="flex items-center space-x-2 mt-2">
                            <Clock className="w-4 h-4 text-slate-400" />
                            <span className="text-sm text-slate-400">
                              {new Date(result.timestamp).toLocaleString()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-slate-400 mb-1">Processing Time</div>
                        <div className="text-2xl font-bold text-blue-400">{result.processing_time.toFixed(3)}s</div>
                      </div>
                    </div>
                    
                    {/* Original Text */}
                    <div className="mb-8 p-6 bg-slate-700/50 rounded-xl border border-slate-600/30">
                      <div className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                          <span className="text-xs text-blue-400">💬</span>
                        </div>
                        <p className="text-white text-lg leading-relaxed">"{result.text}"</p>
                      </div>
                    </div>
                    
                    {/* Top Emotions Section */}
                    <div className="mb-8">
                      <h4 className="text-xl font-semibold text-white mb-4 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-emerald-400" />
                        Top Emotions Detected
                      </h4>
                      
                      <div className="grid md:grid-cols-3 gap-4">
                        {topEmotions.length > 0 ? (
                          topEmotions.map(([emotion, confidence], idx) => {
                            const isPrimary = emotion === result.primary_emotion;
                            const emotionColors = {
                              anger: 'from-red-500 to-red-600',
                              disgust: 'from-orange-500 to-orange-600',
                              fear: 'from-purple-500 to-purple-600',
                              happiness: 'from-yellow-500 to-yellow-600',
                              'no emotion': 'from-gray-500 to-gray-600',
                              sadness: 'from-blue-500 to-blue-600',
                              surprise: 'from-pink-500 to-pink-600'
                            };
                            
                            return (
                              <div key={emotion} className={`relative p-6 rounded-xl border-2 transition-all duration-300 ${
                                isPrimary 
                                  ? 'border-emerald-400 bg-emerald-500/20' 
                                  : 'border-slate-600 bg-slate-700/50'
                              }`}>
                                {isPrimary && (
                                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center">
                                    <span className="text-xs text-white font-bold">1</span>
                                  </div>
                                )}
                                
                                <div className="text-center">
                                  <div className={`w-16 h-16 bg-gradient-to-br ${emotionColors[emotion as keyof typeof emotionColors] || 'from-slate-500 to-slate-600'} rounded-2xl flex items-center justify-center mx-auto mb-4`}>
                                    <span className="text-2xl">
                                      {emotion === 'anger' ? '😠' : 
                                       emotion === 'disgust' ? '🤢' : 
                                       emotion === 'fear' ? '😨' : 
                                       emotion === 'happiness' ? '😊' : 
                                       emotion === 'no emotion' ? '😐' : 
                                       emotion === 'sadness' ? '😢' : 
                                       emotion === 'surprise' ? '😲' : '😐'}
                                    </span>
                                  </div>
                                  <div className="text-lg font-bold text-white mb-2 capitalize">{emotion}</div>
                                  <div className="text-2xl font-bold text-emerald-400">{(confidence * 100).toFixed(1)}%</div>
                                  <div className="text-sm text-slate-400">Confidence</div>
                                </div>
                              </div>
                            );
                          })
                        ) : (
                          <div className="col-span-3 text-center text-slate-400 text-sm py-8">
                            No emotion data available
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Complete Emotion Breakdown */}
                    <div className="mb-8">
                      <h4 className="text-xl font-semibold text-white mb-4 flex items-center">
                        <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                        Complete Emotion Breakdown
                      </h4>
                      
                      <div className="grid grid-cols-7 gap-3">
                        {sortedEmotions.length > 0 ? (
                          sortedEmotions.map(([emotion, confidence]) => (
                            <div key={emotion} className="text-center">
                              <div className="text-xs text-slate-400 mb-2 capitalize">{emotion}</div>
                              <div className="w-full bg-slate-600 rounded-full h-2 mb-2">
                                <div 
                                  className="h-2 rounded-full transition-all duration-300"
                                  style={{ 
                                    width: `${(confidence * 100)}%`,
                                    backgroundColor: getEmotionColor(emotion)
                                  }}
                                ></div>
                              </div>
                              <div className="text-xs text-white">{(confidence * 100).toFixed(0)}%</div>
                            </div>
                          ))
                        ) : (
                          <div className="col-span-7 text-center text-slate-400 text-sm py-4">
                            No emotion data available
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Analysis Summary */}
                    <div className="p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-blue-400/30">
                      <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                        <CheckCircle className="w-5 h-5 mr-2 text-blue-400" />
                        Analysis Summary
                      </h4>
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-emerald-400 capitalize">
                            {result.primary_emotion?.replace('_', ' ') || 'Unknown'}
                          </div>
                          <div className="text-sm text-slate-400">Primary Emotion</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-400">
                            {result.confidence ? (result.confidence * 100).toFixed(1) : '0.0'}%
                          </div>
                          <div className="text-sm text-slate-400">Overall Confidence</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-400">
                            {result.emotions && typeof result.emotions === 'object' ? Object.keys(result.emotions).length : 0}
                          </div>
                          <div className="text-sm text-slate-400">Emotions Detected</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-orange-400">
                            {result.processing_time?.toFixed(3) || '0.000'}s
                          </div>
                          <div className="text-sm text-slate-400">Processing Time</div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              }
            })}
          </div>
        )}
      </div>

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Export Results</h3>
              <button
                onClick={() => setShowExportModal(false)}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Export Format
                </label>
                <select
                  value={selectedExportFormat}
                  onChange={(e) => setSelectedExportFormat(e.target.value)}
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {exportFormats.map((format) => (
                    <option key={format.id} value={format.id}>
                      {format.name || format.id.toUpperCase()} - {format.description}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="bg-slate-700/50 rounded-lg p-3">
                <p className="text-sm text-slate-300">
                  <strong>Exporting:</strong> {results.length} result{results.length !== 1 ? 's' : ''}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Format: {selectedExportFormat.toUpperCase()}
                </p>
              </div>
              
              <div className="flex space-x-3">
                <button
                  onClick={exportResults}
                  disabled={isExporting}
                  className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white py-2 px-4 rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isExporting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Exporting...</span>
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4" />
                      <span>Export</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={() => setShowExportModal(false)}
                  className="px-4 py-2 bg-slate-600 text-slate-300 rounded-lg hover:bg-slate-500 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NewsHeadlines;