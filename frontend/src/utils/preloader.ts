/**
 * Preloader utilities for optimized component loading
 */

type ComponentLoader = () => Promise<{ default: React.ComponentType<Record<string, unknown>> }>;

// Component loaders map
const componentLoaders: Record<string, ComponentLoader> = {
  'ai-hub': () => import('../pages/AIHub'),
  'advanced-optimizer': () => import('../pages/AdvancedOptimizer'),
  'advanced-results': () => import('../pages/AdvancedResults'),
  'current-portfolio-analyzer': () => import('../pages/CurrentPortfolioAnalyzer'),
  'results': () => import('../pages/Results'),
  'index': () => import('../pages/Index'),
};

// Heavy components that should be preloaded
const heavyComponents = [
  'ai-hub',
  'advanced-optimizer',
  'advanced-results'
];

// Cache for loaded components
const componentCache = new Map<string, Promise<{ default: React.ComponentType<Record<string, unknown>> }>>();

/**
 * Preload a component
 */
export const preloadComponent = (componentName: string): Promise<{ default: React.ComponentType<Record<string, unknown>> }> => {
  if (componentCache.has(componentName)) {
    return componentCache.get(componentName)!;
  }

  const loader = componentLoaders[componentName];
  if (!loader) {
    throw new Error(`Component loader not found: ${componentName}`);
  }

  const promise = loader();
  componentCache.set(componentName, promise);
  
  return promise;
};

/**
 * Preload multiple components
 */
export const preloadComponents = async (componentNames: string[]): Promise<void> => {
  const promises = componentNames.map(name => preloadComponent(name));
  await Promise.all(promises);
};

/**
 * Preload heavy components when the app is idle
 */
export const preloadHeavyComponents = (): void => {
  // Use requestIdleCallback for non-critical preloading
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(() => {
      preloadComponents(heavyComponents).catch(console.error);
    });
  } else {
    // Fallback for browsers that don't support requestIdleCallback
    setTimeout(() => {
      preloadComponents(heavyComponents).catch(console.error);
    }, 2000);
  }
};

/**
 * Preload component on mouse hover (for better UX)
 */
export const createHoverPreloader = (componentName: string) => {
  let hasPreloaded = false;
  
  return {
    onMouseEnter: () => {
      if (!hasPreloaded) {
        preloadComponent(componentName).catch(console.error);
        hasPreloaded = true;
      }
    }
  };
};

/**
 * Hook for preloading components based on user interaction
 */
export const usePreloader = () => {
  const preload = (componentName: string) => {
    return preloadComponent(componentName);
  };

  const preloadMultiple = (componentNames: string[]) => {
    return preloadComponents(componentNames);
  };

  const createHoverHandler = (componentName: string) => {
    return createHoverPreloader(componentName);
  };

  return {
    preload,
    preloadMultiple,
    createHoverHandler
  };
};

// Route-based preloading hints
export const routePreloadHints: Record<string, string[]> = {
  '/': ['ai-hub', 'index'], // From homepage, users likely go to AI hub or basic optimizer
  '/basic': ['results', 'advanced-optimizer'], // From basic, users might check results or upgrade
  '/advanced': ['advanced-results'], // From advanced, users will see results
  '/analyze': ['results', 'ai-hub'], // From analyzer, users might see results or AI hub
};

/**
 * Preload components based on current route
 */
export const preloadForRoute = (pathname: string): void => {
  const hints = routePreloadHints[pathname];
  if (hints && hints.length > 0) {
    // Delay preloading to avoid interfering with current page load
    setTimeout(() => {
      preloadComponents(hints).catch(console.error);
    }, 1000);
  }
};

/**
 * Initialize preloader with performance monitoring
 */
export const initializePreloader = (): void => {
  // Monitor performance
  if (typeof performance !== 'undefined' && performance.mark) {
    performance.mark('preloader-start');
  }

  // Start preloading heavy components
  preloadHeavyComponents();

  // Listen for route changes to preload related components
  if (typeof window !== 'undefined') {
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = function(...args) {
      originalPushState.apply(this, args);
      preloadForRoute(location.pathname);
    };

    history.replaceState = function(...args) {
      originalReplaceState.apply(this, args);
      preloadForRoute(location.pathname);
    };

    window.addEventListener('popstate', () => {
      preloadForRoute(location.pathname);
    });

    // Initial preload for current route
    preloadForRoute(location.pathname);
  }

  if (typeof performance !== 'undefined' && performance.mark && performance.measure) {
    performance.mark('preloader-end');
    performance.measure('preloader-init', 'preloader-start', 'preloader-end');
  }
};