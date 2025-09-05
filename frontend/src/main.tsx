import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { initializePreloader } from './utils/preloader';

// Initialize performance monitoring
if (typeof performance !== 'undefined' && performance.mark) {
  performance.mark('app-start');
}

// Initialize preloader for optimized component loading
initializePreloader();

createRoot(document.getElementById("root")!).render(<App />);

// Mark app initialization complete
if (typeof performance !== 'undefined' && performance.mark && performance.measure) {
  performance.mark('app-rendered');
  performance.measure('app-init-time', 'app-start', 'app-rendered');
}
