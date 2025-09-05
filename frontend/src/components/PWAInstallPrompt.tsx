import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Download, Smartphone, Monitor, X, Wifi, WifiOff } from 'lucide-react';
import { usePWA } from '../utils/pwa';

interface PWAInstallPromptProps {
  className?: string;
}

export const PWAInstallPrompt: React.FC<PWAInstallPromptProps> = ({ className = '' }) => {
  const { canInstall, install, isInstalled, isStandalone, isOnline, updateAvailable, updateApp } = usePWA();
  const [showPrompt, setShowPrompt] = useState(false);
  const [showUpdatePrompt, setShowUpdatePrompt] = useState(false);

  useEffect(() => {
    // Show install prompt after user has been on site for 30 seconds
    if (canInstall && !isInstalled) {
      const timer = setTimeout(() => setShowPrompt(true), 30000);
      return () => clearTimeout(timer);
    }
  }, [canInstall, isInstalled]);

  useEffect(() => {
    if (updateAvailable) {
      setShowUpdatePrompt(true);
    }
  }, [updateAvailable]);

  const handleInstall = async () => {
    try {
      await install();
      setShowPrompt(false);
    } catch (error) {
      console.error('Installation failed:', error);
    }
  };

  const handleUpdate = async () => {
    try {
      await updateApp();
      setShowUpdatePrompt(false);
    } catch (error) {
      console.error('Update failed:', error);
    }
  };

  // Don't show anything if app is already installed
  if (isInstalled || isStandalone) {
    return updateAvailable ? (
      <UpdateAvailablePrompt 
        onUpdate={handleUpdate}
        onDismiss={() => setShowUpdatePrompt(false)}
        show={showUpdatePrompt}
      />
    ) : null;
  }

  if (!showPrompt || !canInstall) {
    return (
      <div className={`flex items-center gap-2 text-sm ${className}`}>
        <div className="flex items-center gap-1">
          {isOnline ? (
            <Wifi className="h-4 w-4 text-green-500" />
          ) : (
            <WifiOff className="h-4 w-4 text-orange-500" />
          )}
          <span className={isOnline ? 'text-green-600' : 'text-orange-600'}>
            {isOnline ? 'Online' : 'Offline'}
          </span>
        </div>
      </div>
    );
  }

  return (
    <Card className={`border-primary/20 bg-gradient-to-r from-primary/5 to-secondary/5 ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Download className="h-5 w-5 text-primary" />
            <div>
              <CardTitle className="text-lg">Install Portfolio Optimizer</CardTitle>
              <CardDescription>
                Get the full app experience with offline access
              </CardDescription>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowPrompt(false)}
            aria-label="Dismiss install prompt"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-background/50">
            <Smartphone className="h-6 w-6 text-primary" />
            <div>
              <p className="font-medium text-sm">Mobile Optimized</p>
              <p className="text-xs text-muted-foreground">Fast, native-like experience</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 p-3 rounded-lg bg-background/50">
            <WifiOff className="h-6 w-6 text-primary" />
            <div>
              <p className="font-medium text-sm">Works Offline</p>
              <p className="text-xs text-muted-foreground">Access your data anywhere</p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          <Badge variant="secondary" className="text-xs">
            ⚡ Faster Loading
          </Badge>
          <Badge variant="secondary" className="text-xs">
            📱 Home Screen
          </Badge>
          <Badge variant="secondary" className="text-xs">
            🔔 Push Notifications
          </Badge>
          <Badge variant="secondary" className="text-xs">
            💾 Offline Access
          </Badge>
        </div>

        <div className="flex gap-2">
          <Button onClick={handleInstall} className="flex-1">
            <Download className="h-4 w-4 mr-2" />
            Install App
          </Button>
          <Button 
            variant="outline" 
            onClick={() => setShowPrompt(false)}
          >
            Maybe Later
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

const UpdateAvailablePrompt: React.FC<{
  onUpdate: () => void;
  onDismiss: () => void;
  show: boolean;
}> = ({ onUpdate, onDismiss, show }) => {
  if (!show) return null;

  return (
    <Card className="border-green-200 bg-gradient-to-r from-green-50 to-emerald-50">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center">
              <Download className="h-4 w-4 text-green-600" />
            </div>
            <div>
              <CardTitle className="text-lg text-green-800">Update Available</CardTitle>
              <CardDescription className="text-green-600">
                A new version of Portfolio Optimizer is ready
              </CardDescription>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onDismiss}
            aria-label="Dismiss update prompt"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        <p className="text-sm text-green-700 mb-4">
          This update includes performance improvements, bug fixes, and new features.
        </p>
        
        <div className="flex gap-2">
          <Button onClick={onUpdate} className="bg-green-600 hover:bg-green-700">
            Update Now
          </Button>
          <Button variant="outline" onClick={onDismiss}>
            Later
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

// Connection status indicator
export const ConnectionStatus: React.FC<{ className?: string }> = ({ className = '' }) => {
  const { isOnline } = usePWA();
  
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {isOnline ? (
        <>
          <Wifi className="h-4 w-4 text-green-500" />
          <span className="text-sm text-green-600">Online</span>
        </>
      ) : (
        <>
          <WifiOff className="h-4 w-4 text-orange-500" />
          <span className="text-sm text-orange-600">Offline Mode</span>
        </>
      )}
    </div>
  );
};

export default PWAInstallPrompt;