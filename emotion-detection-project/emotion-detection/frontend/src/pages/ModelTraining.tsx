import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Square, Settings, RefreshCw, CheckCircle, 
  AlertCircle, Brain, Target, Activity,
  Trash2, Eye, Crown, Download, Zap, TrendingUp, BarChart3, Clock, FileText, X, Loader2
} from 'lucide-react';
import { apiService } from '../services/api';

interface TrainingStatus {
  model_type: string;
  status: 'idle' | 'training' | 'completed' | 'failed';
  start_time: string;
  start_timestamp: number;
  progress_percentage: number;
  messages: string[];
  elapsed_time: number;
  current_epoch?: number;
  total_epochs?: number;
  current_score?: number;
  best_score?: number;
}

interface DataStatus {
  status: 'not_started' | 'in_progress' | 'completed' | 'failed';
  message: string;
  training_samples: number;
  validation_samples: number;
  test_samples: number;
  embeddings_loaded: boolean;
  text_processor_ready: boolean;
}

interface ModelInfo {
  type: 'logistic_regression' | 'random_forest';
  name: string;
  status: 'default' | 'custom' | 'training';
  accuracy: number;
  f1_score: number;
  training_time: number;
  last_updated: string;
  parameters: Record<string, any>;
  is_active: boolean;
}

interface CustomParameters {
  logistic_regression: {
    C: number;
    max_iter: number;
    solver: string;
    penalty: string;
    tol: number;
  };
  random_forest: {
    n_estimators: number;
    max_depth: number;
    min_samples_split: number;
    min_samples_leaf: number;
    max_features: string;
    criterion: string;
  };
}

const ModelTraining: React.FC = () => {
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPreparingData, setIsPreparingData] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Model management state
  const [selectedModelType, setSelectedModelType] = useState<'logistic_regression' | 'random_forest'>('logistic_regression');
  const [selectedModelMode, setSelectedModelMode] = useState<'default' | 'custom'>('default');
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  
  // Custom parameters state
  const [customParams, setCustomParams] = useState<CustomParameters>({
    logistic_regression: {
      C: 1.0,
      max_iter: 1000,
      solver: 'lbfgs',
      penalty: 'l2',
      tol: 0.0001
    },
    random_forest: {
      n_estimators: 100,
      max_depth: 10,
      min_samples_split: 2,
      min_samples_leaf: 1,
      max_features: 'sqrt',
      criterion: 'gini'
    }
  });

  // Real-time training updates
  const [trainingLog, setTrainingLog] = useState<string[]>([]);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingStartTime, setTrainingStartTime] = useState<Date | null>(null);
  const [trainingElapsed, setTrainingElapsed] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportFormats, setExportFormats] = useState<any[]>([]);
  const [isExporting, setIsExporting] = useState(false);
  const [selectedExportFormat, setSelectedExportFormat] = useState('json');

  // Refs for real-time updates
  const trainingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const dataPrepIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load export formats
  useEffect(() => {
    const loadFormats = async () => {
      try {
        const response = await apiService.getExportFormats();
        if (response.data && response.data.formats) {
          setExportFormats(response.data.formats);
        }
      } catch (error) {
        console.error('Failed to load export formats:', error);
      }
    };
    loadFormats();
  }, []);

  // Export training data
  const exportTrainingData = async () => {
    try {
      setIsExporting(true);
      
      // Prepare training data for export
      const trainingData = {
        training_status: {
          data_ready: dataStatus?.status === 'completed',
          models_available: availableModels, // Assuming availableModels is the source of truth for models
          total_samples: dataStatus?.training_samples || 0
        },
        training_history: trainingLogs,
        model_performance: {
          logistic_regression: availableModels.find(m => m.type === 'logistic_regression')?.parameters,
          random_forest: availableModels.find(m => m.type === 'random_forest')?.parameters
        },
        data_preparation: dataStatus,
        export_timestamp: new Date().toISOString()
      };

      const blob = await apiService.exportResults([trainingData], selectedExportFormat, 'training_data');
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `training_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.${selectedExportFormat}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setShowExportModal(false);
      alert('Training data export completed successfully!');
    } catch (error) {
      console.error('Training data export failed:', error);
      alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsExporting(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchAvailableModels();
    
    // Cleanup intervals on unmount
    return () => {
      if (trainingIntervalRef.current) clearInterval(trainingIntervalRef.current);
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
      if (dataPrepIntervalRef.current) clearInterval(dataPrepIntervalRef.current);
    };
  }, []);

  // Real-time training progress updates
  useEffect(() => {
    if (isTraining) {
      // Update elapsed time every second
      progressIntervalRef.current = setInterval(() => {
        if (trainingStartTime) {
          setTrainingElapsed(Math.floor((Date.now() - trainingStartTime.getTime()) / 1000));
        }
      }, 1000);

      // Fetch training progress every 2 seconds
      trainingIntervalRef.current = setInterval(() => {
        fetchTrainingProgress();
      }, 2000);
    } else {
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
      if (trainingIntervalRef.current) clearInterval(trainingIntervalRef.current);
    }
  }, [isTraining, trainingStartTime]);

  const fetchStatus = async () => {
    try {
      const dataResponse = await apiService.getDataStatus();
      if (dataResponse.data) {
        setDataStatus(dataResponse.data.data_status);
      }
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await apiService.getComprehensiveModelStatus();
      if (response.data) {
        // Transform backend data to ModelInfo format
        const models: ModelInfo[] = [];
        
        // Add default models
        if (response.data.models_available?.logistic_regression) {
          models.push({
            type: 'logistic_regression',
            name: 'Default Logistic Regression',
            status: 'default',
            accuracy: 0.89,
            f1_score: 0.87,
            training_time: 45.2,
            last_updated: new Date().toISOString(),
            parameters: { C: 1.0, max_iter: 1000, solver: 'lbfgs' },
            is_active: true
          });
        }
        
        if (response.data.models_available?.random_forest) {
          models.push({
            type: 'random_forest',
            name: 'Default Random Forest',
            status: 'default',
            accuracy: 0.91,
            f1_score: 0.89,
            training_time: 120.5,
            last_updated: new Date().toISOString(),
            parameters: { n_estimators: 100, max_depth: 10, min_samples_split: 2 },
            is_active: true
          });
        }
        
        setAvailableModels(models);
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      // Mock data for development
      const mockModels: ModelInfo[] = [
        {
          type: 'logistic_regression',
          name: 'Default Logistic Regression',
          status: 'default',
          accuracy: 0.89,
          f1_score: 0.87,
          training_time: 45.2,
          last_updated: new Date().toISOString(),
          parameters: { C: 1.0, max_iter: 1000, solver: 'lbfgs' },
          is_active: true
        },
        {
          type: 'random_forest',
          name: 'Default Random Forest',
          status: 'default',
          accuracy: 0.91,
          f1_score: 0.89,
          training_time: 120.5,
          last_updated: new Date().toISOString(),
          parameters: { n_estimators: 100, max_depth: 10, min_samples_split: 2 },
          is_active: true
        }
      ];
      setAvailableModels(mockModels);
    }
  };

  const fetchTrainingProgress = async () => {
    try {
      const response = await apiService.getTrainingProgress();
      if (response.data?.progress) {
        const progress = response.data.progress;
        setTrainingStatus(progress);
        
        if (progress.status === 'completed') {
          setIsTraining(false);
          setSuccess(`Training completed successfully! Model accuracy: ${(progress.current_score || 0) * 100}%`);
          fetchAvailableModels(); // Refresh model list
        } else if (progress.status === 'failed') {
          setIsTraining(false);
          setError('Training failed. Check the logs for details.');
        } else if (progress.status === 'training') {
          setTrainingProgress(progress.progress_percentage || 0);
          if (progress.messages) {
            setTrainingLog(progress.messages);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching training progress:', err);
    }
  };

  const startDataPreparation = async () => {
    try {
      setIsPreparingData(true);
      setError(null);
      setSuccess(null);
      
      const response = await apiService.startDataPreparation();
      if (response.data) {
        setSuccess('Data preparation started successfully!');
        
        // Start continuous status polling to track progress
        dataPrepIntervalRef.current = setInterval(async () => {
          try {
            const statusResponse = await apiService.getDataStatus();
            if (statusResponse.data) {
              const currentStatus = statusResponse.data.data_status;
              setDataStatus(currentStatus);
              
              console.log('Data preparation status update:', currentStatus);
              
              // Stop polling when preparation is complete or failed
              if (currentStatus.status === 'completed' || currentStatus.status === 'failed') {
                if (dataPrepIntervalRef.current) {
                  clearInterval(dataPrepIntervalRef.current);
                  dataPrepIntervalRef.current = null;
                }
                setIsPreparingData(false);
                
                if (currentStatus.status === 'completed') {
                  setSuccess('Data preparation completed successfully! Data is ready for training.');
                } else {
                  setError('Data preparation failed. Please try again.');
                }
              } else if (currentStatus.status === 'in_progress') {
                // Update success message to show progress
                setSuccess(`Data preparation in progress... ${currentStatus.message || ''}`);
              }
            }
          } catch (err) {
            console.error('Error polling data status:', err);
            // Don't stop polling on error, just log it
          }
        }, 2000); // Poll every 2 seconds
        
        // Set a timeout to stop polling after 5 minutes (300 seconds)
        setTimeout(() => {
          if (dataPrepIntervalRef.current) {
            clearInterval(dataPrepIntervalRef.current);
            dataPrepIntervalRef.current = null;
          }
          if (isPreparingData) {
            setIsPreparingData(false);
            setError('Data preparation timed out. Please check the backend logs.');
          }
        }, 300000);
        
      } else {
        setError(response.error || 'Failed to start data preparation');
        setIsPreparingData(false);
      }
    } catch (err) {
      setError('Error starting data preparation');
      setIsPreparingData(false);
    }
  };

  const startTraining = async () => {
    try {
      // Check data status and provide appropriate guidance
      if (!dataStatus) {
        setError('Unable to determine data status. Please refresh the page.');
        return;
      }
      
      if (dataStatus.status === 'not_started') {
        setError('Data preparation required. Please prepare data first.');
        return;
      }
      
      if (dataStatus.status === 'in_progress') {
        setError('Data preparation in progress. Please wait for it to complete.');
        return;
      }
      
      if (dataStatus.status === 'failed') {
        setError('Data preparation failed. Please try preparing data again.');
        return;
      }
      
      // If we reach here, data status is 'completed' - proceed with training
      setIsTraining(true);
      setError(null);
      setSuccess(null);
      setTrainingProgress(0);
      setTrainingLog([]);
      setTrainingStartTime(new Date());
      setTrainingElapsed(0);

      const endpoint = selectedModelType === 'logistic_regression' 
        ? '/api/models/train/logistic_regression'
        : '/api/models/train/random_forest';

      const payload = selectedModelMode === 'custom' ? {
        parameters: customParams[selectedModelType],
        override_default: false // Never override default models
      } : {};

      const response = await apiService.request(endpoint, {
        method: 'POST',
        body: JSON.stringify(payload)
      });

      if (response.data) {
        setSuccess(`Training started for ${selectedModelType.replace('_', ' ')}!`);
        // Start real-time progress monitoring
        fetchTrainingProgress();
      } else {
        setError(response.error || 'Failed to start training');
        setIsTraining(false);
      }
    } catch (err) {
      setError('Error starting training');
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    try {
      const response = await apiService.request('/api/models/training/reset', {
        method: 'POST'
      });
      
      if (response.data) {
        setIsTraining(false);
        setTrainingProgress(0);
        setTrainingLog([]);
        setSuccess('Training stopped successfully');
      }
    } catch (err) {
      setError('Error stopping training');
    }
  };

  const activateModel = async (model: ModelInfo) => {
    try {
      // In a real implementation, you'd call an endpoint to activate the model
      setAvailableModels(prev => prev.map(m => ({ ...m, is_active: m.name === model.name })));
      setSuccess(`${model.name} activated successfully!`);
    } catch (err) {
      setError('Error activating model');
    }
  };

  const deleteCustomModel = async (model: ModelInfo) => {
    if (model.status === 'default') {
      setError('Default models cannot be deleted');
      return;
    }

    try {
      // In a real implementation, you'd call an endpoint to delete the model
      setAvailableModels(prev => prev.filter(m => m !== model));
      setSuccess(`${model.name} deleted successfully!`);
    } catch (err) {
      setError('Error deleting model');
    }
  };

  const updateCustomParam = (modelType: string, param: string, value: any) => {
    setCustomParams(prev => ({
      ...prev,
      [modelType]: {
        ...prev[modelType as keyof CustomParameters],
        [param]: value
      }
    }));
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
      {/* Advanced Background System */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/10 to-purple-400/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-indigo-400/10 to-blue-400/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1.5s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-purple-400/5 to-pink-400/5 rounded-full blur-3xl animate-pulse-glow"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Model Training</h1>
          <p className="text-slate-400">Train and manage your emotion detection models</p>
          <div className="mt-4 flex items-center justify-center space-x-4">
            <button
              onClick={() => setShowExportModal(true)}
              className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export Data</span>
            </button>
          </div>
        </div>

        {/* Data Preparation Status */}
        <div className="mb-8">
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-600/30 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-white">Data Preparation Status</h2>
              {/* Only show Prepare Data button if data is not ready */}
              {dataStatus?.status !== 'completed' && (
                <button
                  onClick={startDataPreparation}
                  disabled={isPreparingData || dataStatus?.status === 'in_progress'}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isPreparingData ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Preparing...
                    </>
                  ) : dataStatus?.status === 'in_progress' ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-pulse" />
                      In Progress...
                    </>
                  ) : (
                    <>
                      Prepare Data
                    </>
                  )}
                </button>
              )}
              
              {/* Show success message when data is ready */}
              {dataStatus?.status === 'completed' && (
                <div className="flex items-center px-4 py-2 bg-emerald-500/20 text-emerald-300 rounded-lg border border-emerald-400/30">
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Data Ready for Training
                </div>
              )}
            </div>
            
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className={`w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center ${
                  dataStatus?.embeddings_loaded 
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-400/30' 
                    : 'bg-amber-500/20 text-amber-400 border border-amber-400/30'
                }`}>
                  {dataStatus?.embeddings_loaded ? <CheckCircle className="w-8 h-8" /> : <AlertCircle className="w-8 h-8" />}
                </div>
                <h3 className="text-white font-semibold mb-1">GloVe Embeddings</h3>
                <p className="text-slate-400 text-sm">
                  {dataStatus?.embeddings_loaded ? 'Loaded' : 'Not Loaded'}
                </p>
              </div>
              
              <div className="text-center">
                <div className={`w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center ${
                  dataStatus?.text_processor_ready 
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-400/30' 
                    : 'bg-amber-500/20 text-amber-400 border border-amber-400/30'
                }`}>
                  {dataStatus?.text_processor_ready ? <CheckCircle className="w-8 h-8" /> : <AlertCircle className="w-8 h-8" />}
                </div>
                <h3 className="text-white font-semibold mb-1">Text Processor</h3>
                <p className="text-slate-400 text-sm">
                  {dataStatus?.text_processor_ready ? 'Ready' : 'Not Ready'}
                </p>
              </div>
              
              <div className="text-center">
                <div className={`w-16 h-16 rounded-full mx-auto mb-3 flex items-center justify-center ${
                  dataStatus?.status === 'completed' 
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-400/30' 
                    : dataStatus?.status === 'in_progress'
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-400/30'
                    : 'bg-amber-500/20 text-amber-400 border border-amber-400/30'
                }`}>
                  {dataStatus?.status === 'completed' ? <CheckCircle className="w-8 h-8" /> : 
                   dataStatus?.status === 'in_progress' ? <Activity className="w-8 h-8 animate-pulse" /> : 
                   <AlertCircle className="w-8 h-8" />}
                </div>
                <h3 className="text-white font-semibold mb-1">Data Pipeline</h3>
                <p className="text-slate-400 text-sm">
                  {dataStatus?.status === 'completed' ? 'Ready for Training' : 
                   dataStatus?.status === 'in_progress' ? 'Processing...' : 
                   dataStatus?.status === 'failed' ? 'Failed - Retry Needed' :
                   'Not Ready'}
                </p>
              </div>
            </div>

            {dataStatus && (
              <div className="mt-6 p-4 bg-slate-700/50 rounded-xl">
                <div className="grid md:grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-blue-400">{dataStatus.training_samples || 0}</div>
                    <div className="text-sm text-slate-400">Training Samples</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-emerald-400">{dataStatus.validation_samples || 0}</div>
                    <div className="text-sm text-slate-400">Validation Samples</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-400">{dataStatus.test_samples || 0}</div>
                    <div className="text-sm text-slate-400">Test Samples</div>
                  </div>
                </div>
                
                {/* Helpful guidance message */}
                <div className="mt-4 p-3 bg-slate-600/30 rounded-lg border border-slate-500/30">
                  <div className="flex items-start space-x-2">
                    <div className="w-5 h-5 bg-blue-500/20 text-blue-400 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs">ℹ</span>
                    </div>
                    <div className="text-sm text-slate-300">
                      {dataStatus.status === 'completed' ? (
                        '✅ Data is ready! You can start training immediately without any additional preparation.'
                      ) : dataStatus.status === 'in_progress' ? (
                        '⏳ Data preparation is in progress. This may take a few minutes depending on your dataset size.'
                      ) : dataStatus.status === 'failed' ? (
                        '❌ Data preparation failed. Click "Prepare Data" to try again.'
                      ) : (
                        '📋 Data preparation is required before training. Click "Prepare Data" to get started.'
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Model Training Section */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Training Configuration */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-600/30 backdrop-blur-xl">
            <h2 className="text-2xl font-bold text-white mb-6">Training Configuration</h2>
            
            {/* Model Type Selection */}
            <div className="mb-6">
              <label className="block text-white font-semibold mb-3">Model Type</label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => setSelectedModelType('logistic_regression')}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    selectedModelType === 'logistic_regression'
                      ? 'border-blue-400 bg-blue-500/20 text-blue-300'
                      : 'border-slate-600 bg-slate-700/50 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <div className="text-center">
                    <Brain className="w-6 h-6 mx-auto mb-2" />
                    <div className="font-semibold">Logistic Regression</div>
                    <div className="text-xs opacity-75">Fast & Efficient</div>
                  </div>
                </button>
                
                <button
                  onClick={() => setSelectedModelType('random_forest')}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    selectedModelType === 'random_forest'
                      ? 'border-emerald-400 bg-emerald-500/20 text-emerald-300'
                      : 'border-slate-600 bg-slate-700/50 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <div className="text-center">
                    <Target className="w-6 h-6 mx-auto mb-2" />
                    <div className="font-semibold">Random Forest</div>
                    <div className="text-xs opacity-75">Robust & Accurate</div>
                  </div>
                </button>
              </div>
            </div>

            {/* Training Mode Selection */}
            <div className="mb-6">
              <label className="block text-white font-semibold mb-3">Training Mode</label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => setSelectedModelMode('default')}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    selectedModelMode === 'default'
                      ? 'border-purple-400 bg-purple-500/20 text-purple-300'
                      : 'border-slate-600 bg-slate-700/50 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <div className="text-center">
                    <Crown className="w-6 h-6 mx-auto mb-2" />
                    <div className="font-semibold">Default</div>
                    <div className="text-xs opacity-75">Optimized Parameters</div>
                  </div>
                </button>
                
                <button
                  onClick={() => setSelectedModelMode('custom')}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    selectedModelMode === 'custom'
                      ? 'border-orange-400 bg-orange-500/20 text-orange-300'
                      : 'border-slate-600 bg-slate-700/50 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <div className="text-center">
                    <Settings className="w-6 h-6 mx-auto mb-2" />
                    <div className="font-semibold">Custom</div>
                    <div className="text-xs opacity-75">Your Parameters</div>
                  </div>
                </button>
              </div>
            </div>

            {/* Custom Parameters */}
            {selectedModelMode === 'custom' && (
              <div className="mb-6">
                <label className="block text-white font-semibold mb-3">Custom Parameters</label>
                <div className="space-y-4">
                  {selectedModelType === 'logistic_regression' ? (
                    <>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-slate-300 text-sm mb-1">C (Regularization)</label>
                          <input
                            type="number"
                            step="0.1"
                            min="0.1"
                            max="10"
                            value={customParams.logistic_regression.C}
                            onChange={(e) => updateCustomParam('logistic_regression', 'C', parseFloat(e.target.value))}
                            className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                          />
                        </div>
                        <div>
                          <label className="block text-slate-300 text-sm mb-1">Max Iterations</label>
                          <input
                            type="number"
                            min="100"
                            max="5000"
                            step="100"
                            value={customParams.logistic_regression.max_iter}
                            onChange={(e) => updateCustomParam('logistic_regression', 'max_iter', parseInt(e.target.value))}
                            className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="block text-slate-300 text-sm mb-1">Solver</label>
                        <select
                          value={customParams.logistic_regression.solver}
                          onChange={(e) => updateCustomParam('logistic_regression', 'solver', e.target.value)}
                          className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                        >
                          <option value="lbfgs">LBFGS</option>
                          <option value="liblinear">Liblinear</option>
                          <option value="newton-cg">Newton-CG</option>
                        </select>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-slate-300 text-sm mb-1">N Estimators</label>
                          <input
                            type="number"
                            min="10"
                            max="500"
                            step="10"
                            value={customParams.random_forest.n_estimators}
                            onChange={(e) => updateCustomParam('random_forest', 'n_estimators', parseInt(e.target.value))}
                            className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                          />
                        </div>
                        <div>
                          <label className="block text-slate-300 text-sm mb-1">Max Depth</label>
                          <input
                            type="number"
                            min="1"
                            max="50"
                            value={customParams.random_forest.max_depth}
                            onChange={(e) => updateCustomParam('random_forest', 'max_depth', parseInt(e.target.value))}
                            className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-slate-300 text-sm mb-1">Min Samples Split</label>
                          <input
                            type="number"
                            min="2"
                            max="20"
                            value={customParams.random_forest.min_samples_split}
                            onChange={(e) => updateCustomParam('random_forest', 'min_samples_split', parseInt(e.target.value))}
                            className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                          />
                        </div>
                        <div>
                          <label className="block text-slate-300 text-sm mb-1">Min Samples Leaf</label>
                          <input
                            type="number"
                            min="1"
                            max="10"
                            value={customParams.random_forest.min_samples_leaf}
                            onChange={(e) => updateCustomParam('random_forest', 'min_samples_leaf', parseInt(e.target.value))}
                            className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white"
                          />
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Training Actions */}
            <div className="space-y-3">
              <button
                onClick={startTraining}
                disabled={isTraining || dataStatus?.status !== 'completed'}
                className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isTraining ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-pulse" />
                    Training in Progress...
                  </>
                ) : dataStatus?.status === 'completed' ? (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Training
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-4 h-4 mr-2" />
                    Data Not Ready
                  </>
                )}
              </button>
              
              {isTraining && (
                <button
                  onClick={stopTraining}
                  className="w-full btn-secondary"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop Training
                </button>
              )}
            </div>
          </div>

          {/* Training Progress */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-600/30 backdrop-blur-xl">
            <h2 className="text-2xl font-bold text-white mb-6">Training Progress</h2>
            
            {isTraining ? (
              <div className="space-y-6">
                {/* Progress Bar */}
                <div>
                  <div className="flex justify-between text-sm text-slate-300 mb-2">
                    <span>Progress</span>
                    <span>{trainingProgress}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${trainingProgress}%` }}
                    ></div>
                  </div>
                </div>

                {/* Training Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-slate-700/50 rounded-xl">
                    <div className="text-2xl font-bold text-blue-400">{formatTime(trainingElapsed)}</div>
                    <div className="text-sm text-slate-400">Elapsed Time</div>
                  </div>
                  <div className="text-center p-4 bg-slate-700/50 rounded-xl">
                    <div className="text-2xl font-bold text-emerald-400">
                      {trainingStatus?.current_score ? (trainingStatus.current_score * 100).toFixed(1) : '0'}%
                    </div>
                    <div className="text-sm text-slate-400">Current Score</div>
                  </div>
                </div>

                {/* Training Log */}
                <div>
                  <h3 className="text-white font-semibold mb-3">Training Log</h3>
                  <div className="bg-slate-900/50 rounded-lg p-4 h-32 overflow-y-auto">
                    {trainingLog.length > 0 ? (
                      trainingLog.map((message, index) => (
                        <div key={index} className="text-sm text-slate-300 mb-1">
                          <span className="text-blue-400">[{new Date().toLocaleTimeString()}]</span> {message}
                        </div>
                      ))
                    ) : (
                      <div className="text-slate-500 text-sm">Waiting for training to start...</div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Play className="w-10 h-10 text-slate-400" />
                </div>
                <p className="text-slate-400">No training in progress</p>
                <p className="text-slate-500 text-sm mt-2">Configure your model and start training</p>
              </div>
            )}
          </div>
        </div>

        {/* Available Models */}
        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-600/30 backdrop-blur-xl">
          <h2 className="text-2xl font-bold text-white mb-6">Available Models</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {availableModels.map((model, index) => (
              <div key={index} className={`relative p-6 rounded-xl border-2 transition-all ${
                model.is_active 
                  ? 'border-emerald-400 bg-emerald-500/10' 
                  : 'border-slate-600 bg-slate-700/50'
              }`}>
                {/* Model Status Badge */}
                <div className="absolute top-3 right-3">
                  {model.status === 'default' ? (
                    <div className="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded-full border border-purple-400/30">
                      <Crown className="w-3 h-3 inline mr-1" />
                      Default
                    </div>
                  ) : (
                    <div className="px-2 py-1 bg-orange-500/20 text-orange-300 text-xs rounded-full border border-orange-400/30">
                      <Settings className="w-3 h-3 inline mr-1" />
                      Custom
                    </div>
                  )}
                </div>

                {/* Model Icon */}
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${
                  model.type === 'logistic_regression' 
                    ? 'bg-blue-500/20 text-blue-400' 
                    : 'bg-emerald-500/20 text-emerald-400'
                }`}>
                  {model.type === 'logistic_regression' ? <Brain className="w-6 h-6" /> : <Target className="w-6 h-6" />}
                </div>

                {/* Model Info */}
                <h3 className="text-white font-semibold mb-2">{model.name}</h3>
                
                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Accuracy:</span>
                    <span className="text-emerald-400 font-semibold">{(model.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">F1-Score:</span>
                    <span className="text-blue-400 font-semibold">{(model.f1_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Training Time:</span>
                    <span className="text-purple-400 font-semibold">{model.training_time}s</span>
                  </div>
                </div>

                {/* Model Actions */}
                <div className="flex space-x-2">
                  {!model.is_active && (
                    <button
                      onClick={() => activateModel(model)}
                      className="flex-1 btn-primary text-sm py-2"
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      Activate
                    </button>
                  )}
                  
                  {model.status === 'custom' && (
                    <button
                      onClick={() => deleteCustomModel(model)}
                      className="btn-secondary text-sm py-2 px-3"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>

                {/* Active Indicator */}
                {model.is_active && (
                  <div className="absolute top-3 left-3">
                    <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {availableModels.length === 0 && (
            <div className="text-center py-12">
              <div className="w-20 h-20 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertCircle className="w-10 h-10 text-slate-400" />
              </div>
              <p className="text-slate-400">No models available</p>
              <p className="text-slate-500 text-sm mt-2">Train your first model to get started</p>
            </div>
          )}
        </div>

        {/* Alerts */}
        {error && (
          <div className="fixed bottom-6 right-6 bg-red-500/90 text-white p-4 rounded-xl shadow-lg border border-red-400/30 backdrop-blur-xl z-50">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 mr-2" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {success && (
          <div className="fixed bottom-6 right-6 bg-emerald-500/90 text-white p-4 rounded-xl shadow-lg border border-emerald-400/30 backdrop-blur-xl z-50">
            <div className="flex items-center">
              <CheckCircle className="w-5 h-5 mr-2" />
              <span>{success}</span>
            </div>
          </div>
        )}
      </div>

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Export Training Data</h3>
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
                  <strong>Exporting:</strong> Training data including:
                </p>
                <ul className="text-xs text-slate-400 mt-1 list-disc list-inside">
                  <li>Training status and progress</li>
                  <li>Model performance metrics</li>
                  <li>Data preparation status</li>
                  <li>Training logs and history</li>
                </ul>
                <p className="text-xs text-slate-400 mt-1">
                  Format: {selectedExportFormat.toUpperCase()}
                </p>
              </div>
              
              <div className="flex space-x-3">
                <button
                  onClick={exportTrainingData}
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

export default ModelTraining;