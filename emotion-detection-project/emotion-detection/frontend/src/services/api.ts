const API_BASE_URL = '';

export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: number;
}

class ApiService {
  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          error: data.detail || data.error || 'An error occurred',
          status: response.status,
        };
      }

      return {
        data: data || null,
        status: response.status,
      };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
        status: 0,
      };
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
  async detectEmotion(text: string, modelPreference: string = 'auto'): Promise<ApiResponse> {
    return this.request('/api/detect-emotion', {
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

  // Health Check - consolidated endpoint
  async healthCheck(): Promise<ApiResponse> {
    return this.request('/health');
  }
}

export const apiService = new ApiService();
export default apiService;
