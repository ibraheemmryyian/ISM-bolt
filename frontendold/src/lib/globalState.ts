import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { toast } from 'react-toastify';

// Global state interfaces
interface LoadingState {
  isLoading: boolean;
  loadingMessage: string;
  progress?: number;
}

interface ErrorState {
  hasError: boolean;
  errorMessage: string;
  errorCode?: string;
  retryAction?: () => void;
}

interface UserFeedback {
  showFeedback: boolean;
  feedbackType: 'success' | 'error' | 'warning' | 'info';
  feedbackMessage: string;
  feedbackTitle?: string;
}

interface NetworkState {
  isOnline: boolean;
  lastOnlineCheck: Date;
  connectionQuality: 'excellent' | 'good' | 'poor' | 'offline';
}

interface GlobalState {
  loading: LoadingState;
  error: ErrorState;
  feedback: UserFeedback;
  network: NetworkState;
}

type GlobalAction =
  | { type: 'SET_LOADING'; payload: Partial<LoadingState> }
  | { type: 'CLEAR_LOADING' }
  | { type: 'SET_ERROR'; payload: Partial<ErrorState> }
  | { type: 'CLEAR_ERROR' }
  | { type: 'SHOW_FEEDBACK'; payload: Omit<UserFeedback, 'showFeedback'> }
  | { type: 'HIDE_FEEDBACK' }
  | { type: 'UPDATE_NETWORK'; payload: Partial<NetworkState> }
  | { type: 'SHOW_SUCCESS'; payload: { message: string; title?: string } }
  | { type: 'SHOW_ERROR'; payload: { message: string; title?: string; retryAction?: () => void } }
  | { type: 'SHOW_WARNING'; payload: { message: string; title?: string } }
  | { type: 'SHOW_INFO'; payload: { message: string; title?: string } };

// Initial state
const initialState: GlobalState = {
  loading: {
    isLoading: false,
    loadingMessage: '',
    progress: undefined
  },
  error: {
    hasError: false,
    errorMessage: '',
    errorCode: undefined,
    retryAction: undefined
  },
  feedback: {
    showFeedback: false,
    feedbackType: 'info',
    feedbackMessage: '',
    feedbackTitle: undefined
  },
  network: {
    isOnline: navigator.onLine,
    lastOnlineCheck: new Date(),
    connectionQuality: navigator.onLine ? 'excellent' : 'offline'
  }
};

// Reducer
function globalReducer(state: GlobalState, action: GlobalAction): GlobalState {
  switch (action.type) {
    case 'SET_LOADING':
      return {
        ...state,
        loading: { ...state.loading, ...action.payload }
      };
    
    case 'CLEAR_LOADING':
      return {
        ...state,
        loading: { isLoading: false, loadingMessage: '', progress: undefined }
      };
    
    case 'SET_ERROR':
      return {
        ...state,
        error: { ...state.error, ...action.payload, hasError: true }
      };
    
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: { hasError: false, errorMessage: '', errorCode: undefined, retryAction: undefined }
      };
    
    case 'SHOW_FEEDBACK':
      return {
        ...state,
        feedback: { ...action.payload, showFeedback: true }
      };
    
    case 'HIDE_FEEDBACK':
      return {
        ...state,
        feedback: { ...state.feedback, showFeedback: false }
      };
    
    case 'UPDATE_NETWORK':
      return {
        ...state,
        network: { ...state.network, ...action.payload }
      };
    
    case 'SHOW_SUCCESS':
      toast.success(action.payload.message, { autoClose: 3000 });
      return {
        ...state,
        feedback: {
          showFeedback: true,
          feedbackType: 'success',
          feedbackMessage: action.payload.message,
          feedbackTitle: action.payload.title
        }
      };
    
    case 'SHOW_ERROR':
      toast.error(action.payload.message, { autoClose: 5000 });
      return {
        ...state,
        error: {
          hasError: true,
          errorMessage: action.payload.message,
          errorCode: undefined,
          retryAction: action.payload.retryAction
        },
        feedback: {
          showFeedback: true,
          feedbackType: 'error',
          feedbackMessage: action.payload.message,
          feedbackTitle: action.payload.title
        }
      };
    
    case 'SHOW_WARNING':
      toast.warning(action.payload.message, { autoClose: 4000 });
      return {
        ...state,
        feedback: {
          showFeedback: true,
          feedbackType: 'warning',
          feedbackMessage: action.payload.message,
          feedbackTitle: action.payload.title
        }
      };
    
    case 'SHOW_INFO':
      toast.info(action.payload.message, { autoClose: 3000 });
      return {
        ...state,
        feedback: {
          showFeedback: true,
          feedbackType: 'info',
          feedbackMessage: action.payload.message,
          feedbackTitle: action.payload.title
        }
      };
    
    default:
      return state;
  }
}

// Context
const GlobalStateContext = createContext<{
  state: GlobalState;
  dispatch: React.Dispatch<GlobalAction>;
} | undefined>(undefined);

// Provider component
export function GlobalStateProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(globalReducer, initialState);

  return (
    <GlobalStateContext.Provider value={{ state, dispatch }}>
      {children}
    </GlobalStateContext.Provider>
  );
}

// Hook to use global state
export function useGlobalState() {
  const context = useContext(GlobalStateContext);
  if (context === undefined) {
    throw new Error('useGlobalState must be used within a GlobalStateProvider');
  }
  return context;
}

// Convenience hooks
export function useLoading() {
  const { state } = useGlobalState();
  return state.loading;
}

export function useError() {
  const { state } = useGlobalState();
  return state.error;
}

export function useFeedback() {
  const { state } = useGlobalState();
  return state.feedback;
}

export function useNetwork() {
  const { state } = useGlobalState();
  return state.network;
}

// Action creators
export function useGlobalActions() {
  const { dispatch } = useGlobalState();

  return {
    setLoading: (loading: Partial<LoadingState>) => 
      dispatch({ type: 'SET_LOADING', payload: loading }),
    
    clearLoading: () => 
      dispatch({ type: 'CLEAR_LOADING' }),
    
    setError: (error: Partial<ErrorState>) => 
      dispatch({ type: 'SET_ERROR', payload: error }),
    
    clearError: () => 
      dispatch({ type: 'CLEAR_ERROR' }),
    
    showFeedback: (feedback: Omit<UserFeedback, 'showFeedback'>) => 
      dispatch({ type: 'SHOW_FEEDBACK', payload: feedback }),
    
    hideFeedback: () => 
      dispatch({ type: 'HIDE_FEEDBACK' }),
    
    updateNetworkState: (network: Partial<NetworkState>) => 
      dispatch({ type: 'UPDATE_NETWORK', payload: network }),
    
    showSuccess: (message: string, title?: string) => 
      dispatch({ type: 'SHOW_SUCCESS', payload: { message, title } }),
    
    showError: (message: string, title?: string, retryAction?: () => void) => 
      dispatch({ type: 'SHOW_ERROR', payload: { message, title, retryAction } }),
    
    showWarning: (message: string, title?: string) => 
      dispatch({ type: 'SHOW_WARNING', payload: { message, title } }),
    
    showInfo: (message: string, title?: string) => 
      dispatch({ type: 'SHOW_INFO', payload: { message, title } })
  };
}

// API call wrapper with automatic loading and error handling
export function useApiCall() {
  const { dispatch } = useGlobalState();

  return async <T>(
    apiFunction: () => Promise<T>,
    options: {
      loadingMessage?: string;
      successMessage?: string;
      errorMessage?: string;
      showToast?: boolean;
    } = {}
  ): Promise<T | null> => {
    const {
      loadingMessage = 'Loading...',
      successMessage,
      errorMessage = 'An error occurred. Please try again.',
      showToast = true
    } = options;

    try {
      // Set loading state
      dispatch({ type: 'SET_LOADING', payload: { isLoading: true, loadingMessage } });
      
      // Clear any previous errors
      dispatch({ type: 'CLEAR_ERROR' });
      
      // Execute API call
      const result = await apiFunction();
      
      // Clear loading state
      dispatch({ type: 'CLEAR_LOADING' });
      
      // Show success message if provided
      if (successMessage && showToast) {
        dispatch({ type: 'SHOW_SUCCESS', payload: { message: successMessage } });
      }
      
      return result;
      
    } catch (error: any) {
      // Clear loading state
      dispatch({ type: 'CLEAR_LOADING' });
      
      // Handle error
      const errorMsg = errorMessage || error.message || 'An unexpected error occurred';
      
      if (showToast) {
        dispatch({ type: 'SHOW_ERROR', payload: { message: errorMsg } });
      } else {
        dispatch({ type: 'SET_ERROR', payload: { errorMessage: errorMsg } });
      }
      
      return null;
    }
  };
}

// Network status monitoring
export function initializeNetworkMonitoring() {
  const updateOnlineStatus = () => {
    const isOnline = navigator.onLine;
    const connectionQuality = isOnline ? 'excellent' : 'offline';
    
    // This would need to be called from a component that has access to dispatch
    if (isOnline) {
      toast.success('Connection restored!');
    } else {
      toast.warning('You are currently offline. Some features may not work properly.');
    }
  };

  // Listen for online/offline events
  window.addEventListener('online', updateOnlineStatus);
  window.addEventListener('offline', updateOnlineStatus);
} 