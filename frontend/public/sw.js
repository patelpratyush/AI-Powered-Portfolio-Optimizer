// Service Worker for AI-Powered Portfolio Optimizer
// Provides offline functionality, caching, and background sync

const CACHE_NAME = 'portfolio-optimizer-v1.2.0';
const STATIC_CACHE = 'static-v1.2.0';
const RUNTIME_CACHE = 'runtime-v1.2.0';
const API_CACHE = 'api-v1.2.0';

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
  // Add other critical assets here
];

// API endpoints to cache
const API_ENDPOINTS = [
  '/api/predict/',
  '/api/optimize',
  '/api/cache/status',
  '/api/models/available'
];

// Network-first resources (always try network first)
const NETWORK_FIRST = [
  '/api/predict/',
  '/api/optimize',
  '/api/train/',
  '/api/batch-predict'
];

// Cache-first resources (try cache first)
const CACHE_FIRST = [
  '/api/models/available',
  '/api/cache/status',
  '/api/autocomplete'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE).then((cache) => {
        return cache.addAll(STATIC_ASSETS);
      }),
      caches.open(API_CACHE),
      caches.open(RUNTIME_CACHE)
    ]).then(() => {
      return self.skipWaiting();
    })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          // Delete old caches
          if (cacheName !== STATIC_CACHE && 
              cacheName !== RUNTIME_CACHE && 
              cacheName !== API_CACHE) {
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      return self.clients.claim();
    })
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle different types of requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request));
  } else if (request.destination === 'document') {
    event.respondWith(handleDocumentRequest(request));
  } else if (STATIC_ASSETS.some(asset => url.pathname.endsWith(asset))) {
    event.respondWith(handleStaticAsset(request));
  } else {
    event.respondWith(handleRuntimeCache(request));
  }
});

// API Request Handler
async function handleApiRequest(request) {
  const url = new URL(request.url);
  const cacheName = API_CACHE;

  // Check if this endpoint should be network-first or cache-first
  const isNetworkFirst = NETWORK_FIRST.some(pattern => 
    url.pathname.includes(pattern.replace('/api/', ''))
  );

  const isCacheFirst = CACHE_FIRST.some(pattern => 
    url.pathname.includes(pattern.replace('/api/', ''))
  );

  if (isNetworkFirst) {
    return networkFirst(request, cacheName);
  } else if (isCacheFirst) {
    return cacheFirst(request, cacheName);
  } else {
    // Default to stale-while-revalidate for other API calls
    return staleWhileRevalidate(request, cacheName);
  }
}

// Document Request Handler (for navigation)
async function handleDocumentRequest(request) {
  try {
    const response = await fetch(request);
    const cache = await caches.open(RUNTIME_CACHE);
    cache.put(request, response.clone());
    return response;
  } catch (error) {
    // Serve cached version or fallback
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline page if available
    const offlinePage = await caches.match('/offline.html');
    if (offlinePage) {
      return offlinePage;
    }
    
    // Fallback response
    return new Response(
      createOfflinePage(),
      {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'text/html' }
      }
    );
  }
}

// Static Asset Handler
async function handleStaticAsset(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const response = await fetch(request);
    const cache = await caches.open(STATIC_CACHE);
    cache.put(request, response.clone());
    return response;
  } catch (error) {
    return new Response('Asset not available offline', { status: 404 });
  }
}

// Runtime Cache Handler
async function handleRuntimeCache(request) {
  return staleWhileRevalidate(request, RUNTIME_CACHE);
}

// Caching Strategies
async function networkFirst(request, cacheName) {
  try {
    const response = await fetch(request);
    const cache = await caches.open(cacheName);
    
    // Only cache successful responses
    if (response.status === 200) {
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      // Add offline indicator header
      const headers = new Headers(cachedResponse.headers);
      headers.set('X-Served-By', 'ServiceWorker-Cache');
      return new Response(cachedResponse.body, {
        status: cachedResponse.status,
        statusText: cachedResponse.statusText,
        headers
      });
    }
    throw error;
  }
}

async function cacheFirst(request, cacheName) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    // Update cache in background
    fetch(request).then(response => {
      if (response.status === 200) {
        const cache = caches.open(cacheName);
        cache.then(c => c.put(request, response));
      }
    }).catch(() => {
      // Ignore background update failures
    });
    
    return cachedResponse;
  }
  
  try {
    const response = await fetch(request);
    if (response.status === 200) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    return new Response('Not available offline', { status: 503 });
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cachedResponse = await caches.match(request);
  
  const fetchPromise = fetch(request).then(response => {
    if (response.status === 200) {
      const cache = caches.open(cacheName);
      cache.then(c => c.put(request, response.clone()));
    }
    return response;
  }).catch(() => {
    // Return cached version if network fails
    return cachedResponse;
  });
  
  return cachedResponse || fetchPromise;
}

// Message Handler
self.addEventListener('message', (event) => {
  const { type, payload } = event.data;
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_CACHE_STATS':
      getCacheStats().then(stats => {
        event.ports[0].postMessage({ type: 'CACHE_STATS', payload: stats });
      });
      break;
      
    case 'CLEAR_CACHE':
      clearAllCaches().then(() => {
        event.ports[0].postMessage({ type: 'CACHE_CLEARED' });
      });
      break;
      
    case 'BACKGROUND_SYNC':
      handleBackgroundSync();
      break;
      
    case 'SHOW_NOTIFICATION':
      showNotification(payload.title, payload.options);
      break;
  }
});

// Background Sync Handler
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(handleBackgroundSync());
  }
});

async function handleBackgroundSync() {
  try {
    // Sync offline portfolio data
    const offlineData = await getOfflineData();
    if (offlineData.length > 0) {
      await syncOfflineData(offlineData);
      await clearOfflineData();
      
      // Notify clients
      const clients = await self.clients.matchAll();
      clients.forEach(client => {
        client.postMessage({ type: 'SYNC_COMPLETE' });
      });
    }
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

// Push Notification Handler
self.addEventListener('push', (event) => {
  if (event.data) {
    const data = event.data.json();
    event.waitUntil(
      showNotification(data.title, {
        body: data.body,
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        tag: data.tag || 'portfolio-update',
        requireInteraction: data.requireInteraction || false,
        actions: data.actions || []
      })
    );
  }
});

// Notification Click Handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'view-portfolio') {
    event.waitUntil(
      clients.openWindow('/portfolio')
    );
  } else if (event.action === 'dismiss') {
    // Do nothing, notification is already closed
  } else {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Utility Functions
async function getCacheStats() {
  const cacheNames = await caches.keys();
  const stats = {};
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();
    stats[cacheName] = requests.length;
  }
  
  return stats;
}

async function clearAllCaches() {
  const cacheNames = await caches.keys();
  return Promise.all(
    cacheNames.map(cacheName => caches.delete(cacheName))
  );
}

async function getOfflineData() {
  // This would integrate with IndexedDB or other storage
  // For now, return empty array
  return [];
}

async function syncOfflineData(data) {
  // Sync offline data with server
  for (const item of data) {
    try {
      await fetch('/api/sync', {
        method: 'POST',
        body: JSON.stringify(item),
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      console.error('Failed to sync item:', error);
    }
  }
}

async function clearOfflineData() {
  // Clear synced offline data
  // Implementation depends on storage method
}

async function showNotification(title, options = {}) {
  const defaultOptions = {
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    tag: 'portfolio-optimizer',
    requireInteraction: false
  };
  
  return self.registration.showNotification(title, {
    ...defaultOptions,
    ...options
  });
}

function createOfflinePage() {
  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Portfolio Optimizer - Offline</title>
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          margin: 0;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          text-align: center;
          padding: 20px;
        }
        .container {
          max-width: 400px;
        }
        .icon {
          font-size: 4rem;
          margin-bottom: 1rem;
        }
        h1 {
          font-size: 2rem;
          margin-bottom: 1rem;
        }
        p {
          font-size: 1.1rem;
          line-height: 1.6;
          opacity: 0.9;
        }
        .retry-btn {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
          padding: 12px 24px;
          border-radius: 8px;
          font-size: 1rem;
          cursor: pointer;
          margin-top: 1rem;
          transition: background 0.3s ease;
        }
        .retry-btn:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="icon">📈</div>
        <h1>You're Offline</h1>
        <p>
          Portfolio Optimizer works offline too! Some features may be limited, 
          but you can still view cached data and analysis.
        </p>
        <button class="retry-btn" onclick="window.location.reload()">
          Try Again
        </button>
      </div>
    </body>
    </html>
  `;
}