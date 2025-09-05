/**
 * Progressive Web App utilities and service worker management
 */

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed';
    platform: string;
  }>;
  prompt(): Promise<void>;
}

interface PWAInstallHandler {
  canInstall: boolean;
  install: () => Promise<void>;
  isInstalled: boolean;
  isStandalone: boolean;
}

class PWAManager {
  private deferredPrompt: BeforeInstallPromptEvent | null = null;
  private installPromptShown = false;
  private installCallbacks: Array<(canInstall: boolean) => void> = [];
  private updateCallbacks: Array<() => void> = [];

  constructor() {
    this.init();
  }

  private init() {
    // Listen for install prompt
    window.addEventListener('beforeinstallprompt', (e: Event) => {
      e.preventDefault();
      this.deferredPrompt = e as BeforeInstallPromptEvent;
      this.notifyInstallCallbacks(true);
    });

    // Listen for app installed
    window.addEventListener('appinstalled', () => {
      this.deferredPrompt = null;
      this.notifyInstallCallbacks(false);
      this.trackEvent('pwa_installed');
    });

    // Register service worker
    if ('serviceWorker' in navigator) {
      this.registerServiceWorker();
    }

    // Check for updates periodically
    setInterval(() => this.checkForUpdates(), 60000); // Check every minute
  }

  private async registerServiceWorker() {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/'
      });

      // Listen for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              this.notifyUpdateCallbacks();
            }
          });
        }
      });

      // Handle messages from service worker
      navigator.serviceWorker.addEventListener('message', this.handleServiceWorkerMessage.bind(this));

      console.log('Service Worker registered successfully');
    } catch (error) {
      console.error('Service Worker registration failed:', error);
    }
  }

  private handleServiceWorkerMessage(event: MessageEvent) {
    const { type, payload } = event.data;

    switch (type) {
      case 'CACHE_UPDATED':
        this.notifyUpdateCallbacks();
        break;
      case 'OFFLINE_READY':
        this.showNotification('App is ready to work offline!');
        break;
      case 'SYNC_COMPLETE':
        this.showNotification('Data synchronized successfully');
        break;
    }
  }

  // Public API
  getInstallHandler(): PWAInstallHandler {
    return {
      canInstall: this.deferredPrompt !== null,
      isInstalled: this.isAppInstalled(),
      isStandalone: this.isRunningStandalone(),
      install: this.promptInstall.bind(this)
    };
  }

  async promptInstall(): Promise<void> {
    if (!this.deferredPrompt) {
      throw new Error('App is not installable');
    }

    this.installPromptShown = true;
    await this.deferredPrompt.prompt();
    
    const { outcome } = await this.deferredPrompt.userChoice;
    this.trackEvent('pwa_install_prompt', { outcome });
    
    this.deferredPrompt = null;
    this.notifyInstallCallbacks(false);
  }

  onInstallAvailable(callback: (canInstall: boolean) => void) {
    this.installCallbacks.push(callback);
    // Call immediately if install is already available
    if (this.deferredPrompt) {
      callback(true);
    }
  }

  onUpdateAvailable(callback: () => void) {
    this.updateCallbacks.push(callback);
  }

  async checkForUpdates(): Promise<void> {
    if ('serviceWorker' in navigator) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration) {
        await registration.update();
      }
    }
  }

  async updateApp(): Promise<void> {
    if ('serviceWorker' in navigator) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration && registration.waiting) {
        registration.waiting.postMessage({ type: 'SKIP_WAITING' });
        window.location.reload();
      }
    }
  }

  // Offline capabilities
  isOnline(): boolean {
    return navigator.onLine;
  }

  onConnectionChange(callback: (online: boolean) => void) {
    const handler = () => callback(navigator.onLine);
    window.addEventListener('online', handler);
    window.addEventListener('offline', handler);
  }

  // Data persistence for offline mode
  async storeOfflineData(key: string, data: unknown): Promise<void> {
    try {
      const serializedData = JSON.stringify({
        data,
        timestamp: Date.now(),
        version: '1.0'
      });
      localStorage.setItem(`offline_${key}`, serializedData);
    } catch (error) {
      console.error('Failed to store offline data:', error);
    }
  }

  async getOfflineData<T>(key: string): Promise<T | null> {
    try {
      const item = localStorage.getItem(`offline_${key}`);
      if (!item) return null;
      
      const { data, timestamp } = JSON.parse(item);
      
      // Check if data is stale (older than 24 hours)
      const isStale = Date.now() - timestamp > 24 * 60 * 60 * 1000;
      if (isStale) {
        localStorage.removeItem(`offline_${key}`);
        return null;
      }
      
      return data;
    } catch (error) {
      console.error('Failed to retrieve offline data:', error);
      return null;
    }
  }

  async syncData(): Promise<void> {
    if (!this.isOnline()) {
      console.log('Cannot sync data: offline');
      return;
    }

    // Send sync message to service worker
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({ type: 'BACKGROUND_SYNC' });
    }
  }

  // Push notifications (for future implementation)
  async requestNotificationPermission(): Promise<boolean> {
    if (!('Notification' in window)) {
      console.log('Notifications not supported');
      return false;
    }

    if (Notification.permission === 'granted') {
      return true;
    }

    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }

  showNotification(title: string, options?: NotificationOptions): void {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'SHOW_NOTIFICATION',
        payload: { title, options }
      });
    } else if (Notification.permission === 'granted') {
      new Notification(title, options);
    }
  }

  // Helper methods
  private isAppInstalled(): boolean {
    return window.matchMedia('(display-mode: standalone)').matches ||
           (window.navigator as { standalone?: boolean }).standalone === true ||
           document.referrer.includes('android-app://');
  }

  private isRunningStandalone(): boolean {
    return window.matchMedia('(display-mode: standalone)').matches ||
           (window.navigator as { standalone?: boolean }).standalone === true;
  }

  private notifyInstallCallbacks(canInstall: boolean) {
    this.installCallbacks.forEach(callback => callback(canInstall));
  }

  private notifyUpdateCallbacks() {
    this.updateCallbacks.forEach(callback => callback());
  }

  private trackEvent(eventName: string, properties?: Record<string, unknown>) {
    // Analytics integration point
    if (typeof window !== 'undefined' && (window as { gtag?: (...args: unknown[]) => void }).gtag) {
      (window as { gtag: (...args: unknown[]) => void }).gtag('event', eventName, properties);
    }
    console.log('PWA Event:', eventName, properties);
  }

  // Share API integration
  async sharePortfolio(data: {
    title: string;
    text: string;
    url?: string;
  }): Promise<boolean> {
    if ('share' in navigator) {
      try {
        await navigator.share(data);
        this.trackEvent('portfolio_shared', { method: 'native' });
        return true;
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          console.error('Share failed:', error);
        }
      }
    }
    
    // Fallback to clipboard
    try {
      const shareText = `${data.title}\n${data.text}${data.url ? `\n${data.url}` : ''}`;
      await navigator.clipboard.writeText(shareText);
      this.showNotification('Portfolio data copied to clipboard!');
      this.trackEvent('portfolio_shared', { method: 'clipboard' });
      return true;
    } catch (error) {
      console.error('Clipboard write failed:', error);
      return false;
    }
  }

  // File handling for portfolio import/export
  async handleFileShare(files: FileList): Promise<void> {
    for (const file of Array.from(files)) {
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        try {
          const text = await file.text();
          await this.storeOfflineData('imported_portfolio', {
            filename: file.name,
            content: text,
            type: 'csv'
          });
          
          // Redirect to import page
          window.location.href = '/import?source=share';
        } catch (error) {
          console.error('File handling failed:', error);
        }
      }
    }
  }

  // App shortcuts
  addDynamicShortcuts(shortcuts: Array<{
    name: string;
    url: string;
    description?: string;
  }>) {
    if ('shortcuts' in navigator) {
      (navigator as { shortcuts?: { clear: () => void } }).shortcuts?.clear();
      shortcuts.forEach(shortcut => {
        (navigator as { shortcuts?: { add: (shortcut: { name: string; url: string; description?: string }) => void } }).shortcuts?.add({
          name: shortcut.name,
          url: shortcut.url,
          description: shortcut.description
        });
      });
    }
  }

  // Performance monitoring
  getPerformanceMetrics(): Record<string, number> {
    if ('performance' in window) {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const paint = performance.getEntriesByType('paint');
      
      return {
        loadTime: navigation.loadEventEnd - navigation.loadEventStart,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
        timeToInteractive: navigation.loadEventEnd
      };
    }
    return {};
  }
}

// Create singleton instance
export const pwaManager = new PWAManager();

// React hooks for PWA functionality
export function usePWA() {
  const [installHandler, setInstallHandler] = React.useState(pwaManager.getInstallHandler());
  const [isOnline, setIsOnline] = React.useState(navigator.onLine);
  const [updateAvailable, setUpdateAvailable] = React.useState(false);

  React.useEffect(() => {
    pwaManager.onInstallAvailable((canInstall) => {
      setInstallHandler(pwaManager.getInstallHandler());
    });

    pwaManager.onUpdateAvailable(() => {
      setUpdateAvailable(true);
    });

    pwaManager.onConnectionChange((online) => {
      setIsOnline(online);
    });
  }, []);

  return {
    ...installHandler,
    isOnline,
    updateAvailable,
    updateApp: pwaManager.updateApp.bind(pwaManager),
    sharePortfolio: pwaManager.sharePortfolio.bind(pwaManager),
    storeOfflineData: pwaManager.storeOfflineData.bind(pwaManager),
    getOfflineData: pwaManager.getOfflineData.bind(pwaManager),
    syncData: pwaManager.syncData.bind(pwaManager)
  };
}

export default pwaManager;