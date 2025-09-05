import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Optimize chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunk for large dependencies
          vendor: ['react', 'react-dom'],
          // UI components chunk
          ui: ['@radix-ui/react-accordion', '@radix-ui/react-alert-dialog', 
               '@radix-ui/react-avatar', '@radix-ui/react-button'],
          // Charts and visualization
          charts: ['recharts', 'react-chartjs-2'],
          // Utilities
          utils: ['date-fns', 'clsx', 'class-variance-authority'],
          // React Query and data fetching
          data: ['@tanstack/react-query', 'axios']
        }
      }
    },
    // Enable source maps for better debugging
    sourcemap: mode === 'development',
    // Optimize for production
    minify: mode === 'production' ? 'esbuild' : false,
    // Set chunk size warning limit
    chunkSizeWarningLimit: 1000
  },
  // Performance optimizations
  optimizeDeps: {
    include: [
      'react', 'react-dom', 'react-router-dom',
      '@tanstack/react-query',
      'recharts',
      'date-fns',
      'clsx'
    ]
  }
}));
