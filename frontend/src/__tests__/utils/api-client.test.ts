import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios, { AxiosError } from 'axios';
import { apiClient, isAPIError, getErrorMessage, shouldRetry } from '@/lib/api-client';

// Mock axios
vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      request: vi.fn(),
      interceptors: {
        request: {
          use: vi.fn(),
        },
        response: {
          use: vi.fn(),
        },
      },
    })),
  },
}));

const mockAxiosInstance = {
  request: vi.fn(),
  interceptors: {
    request: {
      use: vi.fn(),
    },
    response: {
      use: vi.fn(),
    },
  },
};

describe('APIClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (axios.create as vi.Mock).mockReturnValue(mockAxiosInstance);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('successful requests', () => {
    it('handles GET requests successfully', async () => {
      const mockResponse = {
        data: { test: 'data' },
        status: 200,
      };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const result = await apiClient.get('/test-endpoint');

      expect(result).toEqual({
        data: { test: 'data' },
        status: 200,
        success: true,
      });
      expect(mockAxiosInstance.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/test-endpoint',
      });
    });

    it('handles POST requests successfully', async () => {
      const mockResponse = {
        data: { id: 1, name: 'Test' },
        status: 201,
      };
      const postData = { name: 'Test', value: 123 };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const result = await apiClient.post('/test-endpoint', postData);

      expect(result).toEqual({
        data: { id: 1, name: 'Test' },
        status: 201,
        success: true,
      });
      expect(mockAxiosInstance.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/test-endpoint',
        data: postData,
      });
    });

    it('handles PUT requests successfully', async () => {
      const mockResponse = {
        data: { updated: true },
        status: 200,
      };
      const putData = { id: 1, name: 'Updated' };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const result = await apiClient.put('/test-endpoint', putData);

      expect(result.success).toBe(true);
      expect(result.data).toEqual({ updated: true });
    });

    it('handles DELETE requests successfully', async () => {
      const mockResponse = {
        data: null,
        status: 204,
      };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const result = await apiClient.delete('/test-endpoint');

      expect(result.success).toBe(true);
      expect(result.status).toBe(204);
    });
  });

  describe('error handling', () => {
    it('handles network errors', async () => {
      const networkError = new Error('Network Error') as AxiosError;
      networkError.code = 'NETWORK_ERROR';
      mockAxiosInstance.request.mockRejectedValue(networkError);

      const result = await apiClient.get('/test-endpoint');

      expect(result.success).toBe(false);
      expect(result.error?.error).toBe('NetworkError');
      expect(result.error?.message).toContain('Unable to connect to the server');
    });

    it('handles timeout errors', async () => {
      const timeoutError = new Error('Timeout') as AxiosError;
      timeoutError.code = 'TIMEOUT';
      mockAxiosInstance.request.mockRejectedValue(timeoutError);

      const result = await apiClient.get('/test-endpoint');

      expect(result.success).toBe(false);
      expect(result.error?.error).toBe('TimeoutError');
      expect(result.error?.message).toContain('Request timed out');
    });

    it('handles HTTP error responses', async () => {
      const httpError = new Error('Bad Request') as AxiosError;
      httpError.response = {
        status: 400,
        data: {
          error: 'ValidationError',
          message: 'Invalid input data',
          details: { field: 'required' },
        },
        statusText: 'Bad Request',
        headers: {},
        config: {},
      };
      mockAxiosInstance.request.mockRejectedValue(httpError);

      const result = await apiClient.post('/test-endpoint', {});

      expect(result.success).toBe(false);
      expect(result.error?.error).toBe('ValidationError');
      expect(result.error?.message).toBe('Invalid input data');
      expect(result.error?.details).toEqual({ field: 'required' });
      expect(result.status).toBe(400);
    });

    it('handles unknown errors gracefully', async () => {
      const unknownError = new Error('Unknown error') as AxiosError;
      mockAxiosInstance.request.mockRejectedValue(unknownError);

      const result = await apiClient.get('/test-endpoint');

      expect(result.success).toBe(false);
      expect(result.error?.error).toBe('UnknownError');
      expect(result.error?.message).toBe('An unexpected error occurred');
    });
  });

  describe('specialized methods', () => {
    it('optimizePortfolio method works correctly', async () => {
      const mockResponse = {
        data: { 
          weights: { AAPL: 0.4, GOOGL: 0.6 },
          expected_return: 0.12,
          volatility: 0.15 
        },
        status: 200,
      };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const portfolioData = {
        tickers: ['AAPL', 'GOOGL'],
        strategy: 'sharpe',
        start_date: '2023-01-01',
        end_date: '2024-01-01',
      };

      const result = await apiClient.optimizePortfolio(portfolioData);

      expect(result.success).toBe(true);
      expect(result.data).toHaveProperty('weights');
      expect(mockAxiosInstance.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/optimize',
        data: portfolioData,
      });
    });

    it('predictStock method works correctly', async () => {
      const mockResponse = {
        data: {
          predictions: [
            { date: '2024-01-01', predicted_price: 150.0 }
          ]
        },
        status: 200,
      };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const result = await apiClient.predictStock('AAPL', { days: 5, models: 'ensemble' });

      expect(result.success).toBe(true);
      expect(mockAxiosInstance.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/predict/AAPL?days=5&models=ensemble',
      });
    });

    it('batchPredict method works correctly', async () => {
      const mockResponse = {
        data: {
          predictions: {
            AAPL: [{ date: '2024-01-01', predicted_price: 150.0 }],
            GOOGL: [{ date: '2024-01-01', predicted_price: 2800.0 }]
          }
        },
        status: 200,
      };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const batchData = {
        tickers: ['AAPL', 'GOOGL'],
        days: 10,
        models: 'all',
      };

      const result = await apiClient.batchPredict(batchData);

      expect(result.success).toBe(true);
      expect(mockAxiosInstance.request).toHaveBeenCalledWith({
        method: 'POST',
        url: '/batch-predict',
        data: batchData,
      });
    });

    it('searchTickers method works correctly', async () => {
      const mockResponse = {
        data: {
          suggestions: [
            { symbol: 'AAPL', name: 'Apple Inc.' },
            { symbol: 'AMZN', name: 'Amazon.com Inc.' }
          ]
        },
        status: 200,
      };
      mockAxiosInstance.request.mockResolvedValue(mockResponse);

      const result = await apiClient.searchTickers('A');

      expect(result.success).toBe(true);
      expect(mockAxiosInstance.request).toHaveBeenCalledWith({
        method: 'GET',
        url: '/autocomplete?q=A',
      });
    });
  });
});

describe('utility functions', () => {
  describe('isAPIError', () => {
    it('identifies API errors correctly', () => {
      const errorResponse = {
        success: false,
        status: 400,
        error: {
          error: 'ValidationError',
          message: 'Invalid input',
        },
      };

      expect(isAPIError(errorResponse)).toBe(true);
    });

    it('identifies successful responses correctly', () => {
      const successResponse = {
        success: true,
        status: 200,
        data: { test: 'data' },
      };

      expect(isAPIError(successResponse)).toBe(false);
    });
  });

  describe('getErrorMessage', () => {
    it('extracts error message from API error', () => {
      const errorResponse = {
        success: false,
        status: 400,
        error: {
          error: 'ValidationError',
          message: 'Invalid input data',
        },
      };

      expect(getErrorMessage(errorResponse)).toBe('Invalid input data');
    });

    it('falls back to error type when message is missing', () => {
      const errorResponse = {
        success: false,
        status: 400,
        error: {
          error: 'ValidationError',
        },
      };

      expect(getErrorMessage(errorResponse)).toBe('ValidationError');
    });

    it('returns default message for unknown errors', () => {
      const errorResponse = {
        success: false,
        status: 500,
      };

      expect(getErrorMessage(errorResponse)).toBe('Unknown error');
    });
  });

  describe('shouldRetry', () => {
    it('identifies retryable server errors', () => {
      const serverErrorResponse = {
        success: false,
        status: 500,
        error: {
          error: 'InternalServerError',
          status: 500,
        },
      };

      expect(shouldRetry(serverErrorResponse)).toBe(true);
    });

    it('identifies retryable timeout errors', () => {
      const timeoutResponse = {
        success: false,
        status: 408,
        error: {
          error: 'TimeoutError',
          status: 408,
        },
      };

      expect(shouldRetry(timeoutResponse)).toBe(true);
    });

    it('identifies retryable rate limit errors', () => {
      const rateLimitResponse = {
        success: false,
        status: 429,
        error: {
          error: 'RateLimitExceeded',
          status: 429,
        },
      };

      expect(shouldRetry(rateLimitResponse)).toBe(true);
    });

    it('identifies non-retryable client errors', () => {
      const clientErrorResponse = {
        success: false,
        status: 400,
        error: {
          error: 'ValidationError',
          status: 400,
        },
      };

      expect(shouldRetry(clientErrorResponse)).toBe(false);
    });

    it('identifies non-retryable authentication errors', () => {
      const authErrorResponse = {
        success: false,
        status: 401,
        error: {
          error: 'Unauthorized',
          status: 401,
        },
      };

      expect(shouldRetry(authErrorResponse)).toBe(false);
    });
  });
});