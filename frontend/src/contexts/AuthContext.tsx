import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { apiClient } from '@/lib/api-client';

interface User {
  id: number;
  email: string;
  first_name: string;
  last_name: string;
  preferred_currency: string;
  risk_tolerance: number;
  investment_horizon: string;
}

interface AuthContextType {
  user: User | null;
  tokens: {
    access_token: string;
    refresh_token: string;
    token_type: string;
  } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (userData: RegisterData) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  updateProfile: (updates: Partial<User>) => Promise<{ success: boolean; error?: string }>;
  refreshTokens: () => Promise<boolean>;
}

interface RegisterData {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  preferred_currency?: string;
  risk_tolerance?: number;
  investment_horizon?: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthContextType['tokens']>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load auth data from localStorage on mount
  useEffect(() => {
    const loadAuthData = async () => {
      try {
        const storedTokens = localStorage.getItem('auth_tokens');
        const storedUser = localStorage.getItem('auth_user');

        if (storedTokens && storedUser) {
          const parsedTokens = JSON.parse(storedTokens);
          const parsedUser = JSON.parse(storedUser);

          setTokens(parsedTokens);
          setUser(parsedUser);

          // Verify token is still valid
          const response = await apiClient.get('/auth/me', {
            headers: {
              Authorization: `Bearer ${parsedTokens.access_token}`,
            },
          });

          if (response.success) {
            setUser(response.data.user);
          } else {
            // Token invalid, try to refresh
            const refreshed = await refreshTokens();
            if (!refreshed) {
              clearAuthData();
            }
          }
        }
      } catch (error) {
        console.error('Error loading auth data:', error);
        clearAuthData();
      } finally {
        setIsLoading(false);
      }
    };

    loadAuthData();
  }, [refreshTokens]);

  const clearAuthData = () => {
    setUser(null);
    setTokens(null);
    localStorage.removeItem('auth_tokens');
    localStorage.removeItem('auth_user');
  };

  const saveAuthData = (userData: User, tokenData: AuthContextType['tokens']) => {
    setUser(userData);
    setTokens(tokenData);
    localStorage.setItem('auth_user', JSON.stringify(userData));
    localStorage.setItem('auth_tokens', JSON.stringify(tokenData));
  };

  const login = async (email: string, password: string) => {
    try {
      setIsLoading(true);
      const response = await apiClient.post('/auth/login', {
        email,
        password,
      });

      if (response.success && response.data) {
        const { user: userData, tokens: tokenData } = response.data;
        saveAuthData(userData, tokenData);
        return { success: true };
      } else {
        return {
          success: false,
          error: response.error?.message || 'Login failed',
        };
      }
    } catch (error) {
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterData) => {
    try {
      setIsLoading(true);
      const response = await apiClient.post('/auth/register', userData);

      if (response.success && response.data) {
        const { user: newUser, tokens: tokenData } = response.data;
        saveAuthData(newUser, tokenData);
        return { success: true };
      } else {
        return {
          success: false,
          error: response.error?.message || 'Registration failed',
        };
      }
    } catch (error) {
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      if (tokens?.access_token) {
        await apiClient.post('/auth/logout', {}, {
          headers: {
            Authorization: `Bearer ${tokens.access_token}`,
          },
        });
      }
    } catch (error) {
      console.error('Error during logout:', error);
    } finally {
      clearAuthData();
    }
  };

  const updateProfile = async (updates: Partial<User>) => {
    try {
      if (!tokens?.access_token) {
        return { success: false, error: 'Not authenticated' };
      }

      const response = await apiClient.put('/auth/me', updates, {
        headers: {
          Authorization: `Bearer ${tokens.access_token}`,
        },
      });

      if (response.success && response.data) {
        const updatedUser = response.data.user;
        setUser(updatedUser);
        localStorage.setItem('auth_user', JSON.stringify(updatedUser));
        return { success: true };
      } else {
        return {
          success: false,
          error: response.error?.message || 'Profile update failed',
        };
      }
    } catch (error) {
      return {
        success: false,
        error: 'Network error. Please try again.',
      };
    }
  };

  const refreshTokens = useCallback(async (): Promise<boolean> => {
    try {
      if (!tokens?.refresh_token) {
        return false;
      }

      const response = await apiClient.post('/auth/refresh', {}, {
        headers: {
          Authorization: `Bearer ${tokens.refresh_token}`,
        },
      });

      if (response.success && response.data) {
        const newTokens = {
          ...tokens,
          access_token: response.data.access_token,
        };
        setTokens(newTokens);
        localStorage.setItem('auth_tokens', JSON.stringify(newTokens));
        return true;
      } else {
        clearAuthData();
        return false;
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
      clearAuthData();
      return false;
    }
  }, [tokens]);

  // Set up token refresh interceptor
  useEffect(() => {
    const setupTokenRefresh = () => {
      if (!tokens?.access_token) return;

      // You would implement this in your API client
      // This is a placeholder for the actual implementation
    };

    setupTokenRefresh();
  }, [tokens]);

  const value: AuthContextType = {
    user,
    tokens,
    isAuthenticated: !!user && !!tokens,
    isLoading,
    login,
    register,
    logout,
    updateProfile,
    refreshTokens,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}