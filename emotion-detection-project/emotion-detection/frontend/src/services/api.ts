// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 
  (window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'https://emotion-detection-backend-r8t4.onrender.com');

export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: number;
}

class ApiService {
  async request<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      console.log('Making API request to:', url, 'with options:', options);
      
      // Ensure proper headers for POST requests
      if (options.method === 'POST' && !options.headers) {
        options.headers = {
          'Content-Type': 'application/json',
        };
      }
      
      const response = await fetch(url, {
        ...options,
      });

      const data = await response.json();
      console.log('API response:', { status: response.status, data });
      
      if (!response.ok) {
        console.error('API error response:', { status: response.status, data });
        return {
          data: undefined,
          status: response.status,
          error: data?.detail || `HTTP ${response.status}: ${response.statusText}`
        };
      }

      return {
        data: data || undefined,
        status: response.status,
      };
    } catch (error) {
      return {
        data: undefined,
        status: 500,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async requestBlob(endpoint: string, options: RequestInit = {}): Promise<Blob> {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      
      const response = await fetch(url, {
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData?.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.blob();
    } catch (error) {
      throw new Error(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Model Training
  async startRandomForestTraining(): Promise<ApiResponse> {
    return this.request('/api/models/train/random_forest', {
      method: 'POST',
    });
  }

  async startLogisticRegressionTraining(): Promise<ApiResponse> {
    return this.request('/api/models/train/logistic_regression', {
      method: 'POST',
    });
  }

  async startCustomTraining(modelType: string, parameters: any): Promise<ApiResponse> {
    return this.request(`/api/models/train/${modelType}`, {
      method: 'POST',
      body: JSON.stringify({ parameters, override_default: false }),
    });
  }

  async getTrainingProgress(): Promise<ApiResponse> {
    return this.request('/api/models/training/progress');
  }

  // Data Preparation
  async getDataStatus(): Promise<ApiResponse> {
    return this.request('/api/models/data/status');
  }

  async startDataPreparation(): Promise<ApiResponse> {
    return this.request('/api/models/data/prepare', {
      method: 'POST',
    });
  }

  // Model Status
  async getModelStatus(): Promise<ApiResponse> {
    return this.request('/api/models/status');
  }

  async compareModels(): Promise<ApiResponse> {
    return this.request('/api/models/compare');
  }

  async getComprehensiveModelStatus(): Promise<ApiResponse> {
    return this.request('/api/models/status/comprehensive');
  }

  async evaluateModelsOnTest(): Promise<ApiResponse> {
    return this.request('/api/models/evaluate/test');
  }

  // Emotion Detection
  async detectEmotion(text: string, modelPreference: string = 'logistic_regression'): Promise<ApiResponse> {
    return this.request('/api/detect-emotion', {
      method: 'POST',
      body: JSON.stringify({ text, model_preference: modelPreference }),
    });
  }

  async detectEmotionMultiline(text: string, modelPreference: string = 'logistic_regression'): Promise<ApiResponse> {
    return this.request('/api/detect-emotion/multiline', {
      method: 'POST',
      body: JSON.stringify({ text, model_preference: modelPreference }),
    });
  }

  // System Status - now consolidated into health endpoint
  // async getSystemStatus(): Promise<ApiResponse> {
  //   return this.request('/api/status');
  // }

  async getDatasetInfo(): Promise<ApiResponse> {
    return this.request('/api/dataset/info');
  }

  async getLiveNews(): Promise<ApiResponse> {
    return this.request('/api/news/live');
  }

  async forceRefreshNews(): Promise<ApiResponse> {
    return this.request('/api/news/refresh', {
      method: 'POST',
    });
  }

  // Export functionality
  async getExportFormats(): Promise<ApiResponse> {
    return this.request('/api/export/formats');
  }

  async exportResults(results: any[], format: string, filename?: string): Promise<Blob> {
    return this.requestBlob('/api/export/results', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        results,
        format,
        filename: filename || `emotion_analysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}`
      }),
    });
  }

  // Health Check - consolidated endpoint
  async healthCheck(): Promise<ApiResponse> {
    return this.request('/health');
  }
}

export const apiService = new ApiService();
export default apiService;
