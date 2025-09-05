/**
 * WCAG 2.1 AA Accessibility Utilities
 * Compliance utilities for meeting 2025 legal requirements
 */

/**
 * Color contrast utilities for WCAG AA compliance
 */
export const ColorContrast = {
  // WCAG AA requires 4.5:1 contrast ratio for normal text
  // WCAG AA requires 3:1 contrast ratio for large text (18pt+ or 14pt+ bold)
  
  /**
   * Calculate relative luminance of a color
   */
  getLuminance(color: string): number {
    const rgb = this.hexToRgb(color);
    if (!rgb) return 0;
    
    const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(c => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  },
  
  /**
   * Calculate contrast ratio between two colors
   */
  getContrastRatio(color1: string, color2: string): number {
    const l1 = this.getLuminance(color1);
    const l2 = this.getLuminance(color2);
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    return (lighter + 0.05) / (darker + 0.05);
  },
  
  /**
   * Check if color combination meets WCAG AA standards
   */
  meetsWCAG_AA(foreground: string, background: string, isLargeText = false): boolean {
    const ratio = this.getContrastRatio(foreground, background);
    return isLargeText ? ratio >= 3 : ratio >= 4.5;
  },
  
  /**
   * Check if color combination meets WCAG AAA standards
   */
  meetsWCAG_AAA(foreground: string, background: string, isLargeText = false): boolean {
    const ratio = this.getContrastRatio(foreground, background);
    return isLargeText ? ratio >= 4.5 : ratio >= 7;
  },
  
  /**
   * Convert hex color to RGB
   */
  hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  }
};

/**
 * Focus management utilities
 */
export const FocusManagement = {
  /**
   * Set focus to element with proper error handling
   */
  setFocus(element: HTMLElement | null, options?: FocusOptions): void {
    if (element && typeof element.focus === 'function') {
      try {
        element.focus(options);
      } catch (error) {
        console.warn('Failed to set focus:', error);
      }
    }
  },
  
  /**
   * Get all focusable elements within a container
   */
  getFocusableElements(container: HTMLElement): HTMLElement[] {
    const focusableSelectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable]',
      'audio[controls]',
      'video[controls]'
    ].join(', ');
    
    return Array.from(container.querySelectorAll(focusableSelectors))
      .filter((el): el is HTMLElement => {
        const htmlEl = el as HTMLElement;
        return htmlEl.offsetWidth > 0 && 
               htmlEl.offsetHeight > 0 && 
               !htmlEl.hidden &&
               getComputedStyle(htmlEl).visibility !== 'hidden';
      });
  },
  
  /**
   * Create a focus trap for modals/dialogs
   */
  createFocusTrap(container: HTMLElement): () => void {
    const focusableElements = this.getFocusableElements(container);
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];
    
    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      
      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        this.setFocus(lastElement);
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        this.setFocus(firstElement);
      }
    };
    
    container.addEventListener('keydown', handleTabKey);
    this.setFocus(firstElement);
    
    return () => {
      container.removeEventListener('keydown', handleTabKey);
    };
  }
};

/**
 * Keyboard navigation utilities
 */
export const KeyboardNavigation = {
  /**
   * Handle arrow key navigation in lists/grids
   */
  handleArrowKeys(e: KeyboardEvent, items: HTMLElement[], currentIndex: number): number | null {
    let newIndex = null;
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        newIndex = (currentIndex + 1) % items.length;
        break;
      case 'ArrowUp':
        e.preventDefault();
        newIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = items.length - 1;
        break;
    }
    
    if (newIndex !== null) {
      FocusManagement.setFocus(items[newIndex]);
    }
    
    return newIndex;
  },
  
  /**
   * Handle escape key for closing modals/dropdowns
   */
  handleEscapeKey(e: KeyboardEvent, callback: () => void): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      callback();
    }
  }
};

/**
 * ARIA utilities for screen readers
 */
export const AriaUtils = {
  /**
   * Announce message to screen readers
   */
  announce(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    const announcer = document.createElement('div');
    announcer.setAttribute('aria-live', priority);
    announcer.setAttribute('aria-atomic', 'true');
    announcer.className = 'sr-only';
    announcer.textContent = message;
    
    document.body.appendChild(announcer);
    
    setTimeout(() => {
      document.body.removeChild(announcer);
    }, 1000);
  },
  
  /**
   * Generate unique ID for ARIA relationships
   */
  generateId(prefix = 'aria'): string {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  },
  
  /**
   * Set ARIA expanded state
   */
  setExpanded(element: HTMLElement, expanded: boolean): void {
    element.setAttribute('aria-expanded', expanded.toString());
  },
  
  /**
   * Set ARIA selected state
   */
  setSelected(element: HTMLElement, selected: boolean): void {
    element.setAttribute('aria-selected', selected.toString());
  }
};

/**
 * Text and content accessibility utilities
 */
export const ContentAccessibility = {
  /**
   * Check if text size meets minimum requirements
   */
  isTextSizeAccessible(fontSize: number): boolean {
    // WCAG recommends minimum 12px, but 16px is better for accessibility
    return fontSize >= 16;
  },
  
  /**
   * Generate descriptive text for complex UI elements
   */
  generateDescription(element: { type: string; value?: string | number; label?: string }): string {
    const { type, value, label } = element;
    
    switch (type) {
      case 'chart':
        return `Interactive chart showing ${label}. Use arrow keys to navigate data points.`;
      case 'percentage':
        return `${label}: ${value}%`;
      case 'currency':
        return `${label}: ${value}`;
      case 'button':
        return `Button: ${label}. Press Enter or Space to activate.`;
      default:
        return label || 'Interactive element';
    }
  },
  
  /**
   * Check if alt text is meaningful
   */
  isMeaningfulAltText(alt: string): boolean {
    if (!alt || alt.trim().length === 0) return false;
    
    const meaninglessTexts = ['image', 'photo', 'picture', 'img', 'graphic'];
    const lowerAlt = alt.toLowerCase().trim();
    
    return !meaninglessTexts.some(text => lowerAlt === text);
  }
};

/**
 * Reduced motion utilities
 */
export const ReducedMotion = {
  /**
   * Check if user prefers reduced motion
   */
  prefersReducedMotion(): boolean {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  },
  
  /**
   * Get appropriate animation duration based on user preference
   */
  getAnimationDuration(normal: number, reduced = 0): number {
    return this.prefersReducedMotion() ? reduced : normal;
  },
  
  /**
   * Conditionally apply animation class
   */
  getAnimationClass(normalClass: string, reducedClass = ''): string {
    return this.prefersReducedMotion() ? reducedClass : normalClass;
  }
};

/**
 * High contrast mode utilities
 */
export const HighContrast = {
  /**
   * Check if high contrast mode is enabled
   */
  isHighContrastMode(): boolean {
    return window.matchMedia('(prefers-contrast: high)').matches;
  },
  
  /**
   * Get appropriate styles for high contrast mode
   */
  getHighContrastStyles(normalStyles: React.CSSProperties, highContrastStyles: React.CSSProperties): React.CSSProperties {
    return this.isHighContrastMode() ? { ...normalStyles, ...highContrastStyles } : normalStyles;
  }
};

/**
 * Form accessibility utilities
 */
export const FormAccessibility = {
  /**
   * Generate proper error message ID for form fields
   */
  getErrorId(fieldId: string): string {
    return `${fieldId}-error`;
  },
  
  /**
   * Generate proper description ID for form fields
   */
  getDescriptionId(fieldId: string): string {
    return `${fieldId}-description`;
  },
  
  /**
   * Get ARIA attributes for form field with error
   */
  getFieldAriaAttributes(fieldId: string, hasError: boolean, hasDescription = false) {
    const attributes: Record<string, string> = {};
    
    if (hasError) {
      attributes['aria-invalid'] = 'true';
      attributes['aria-describedby'] = this.getErrorId(fieldId);
    }
    
    if (hasDescription && !hasError) {
      attributes['aria-describedby'] = this.getDescriptionId(fieldId);
    }
    
    if (hasDescription && hasError) {
      attributes['aria-describedby'] = `${this.getDescriptionId(fieldId)} ${this.getErrorId(fieldId)}`;
    }
    
    return attributes;
  }
};

/**
 * Accessibility testing utilities (for development)
 */
export const AccessibilityTesting = {
  /**
   * Check for common accessibility issues
   */
  runBasicChecks(): Array<{ issue: string; element?: HTMLElement; severity: 'error' | 'warning' }> {
    const issues: Array<{ issue: string; element?: HTMLElement; severity: 'error' | 'warning' }> = [];
    
    // Check for images without alt text
    const images = document.querySelectorAll('img:not([alt])');
    images.forEach(img => {
      issues.push({
        issue: 'Image missing alt attribute',
        element: img as HTMLElement,
        severity: 'error'
      });
    });
    
    // Check for buttons without accessible names
    const buttons = document.querySelectorAll('button:not([aria-label]):not([aria-labelledby])');
    buttons.forEach(button => {
      if (!button.textContent?.trim()) {
        issues.push({
          issue: 'Button without accessible name',
          element: button as HTMLElement,
          severity: 'error'
        });
      }
    });
    
    // Check for form inputs without labels
    const inputs = document.querySelectorAll('input:not([type="hidden"]):not([aria-label]):not([aria-labelledby])');
    inputs.forEach(input => {
      const inputEl = input as HTMLInputElement;
      const label = document.querySelector(`label[for="${inputEl.id}"]`);
      if (!label && !inputEl.closest('label')) {
        issues.push({
          issue: 'Form input without label',
          element: inputEl,
          severity: 'error'
        });
      }
    });
    
    // Check for insufficient color contrast (basic check)
    const elements = document.querySelectorAll('*');
    elements.forEach(element => {
      const el = element as HTMLElement;
      const styles = getComputedStyle(el);
      const color = styles.color;
      const backgroundColor = styles.backgroundColor;
      
      if (color && backgroundColor && color !== 'rgba(0, 0, 0, 0)' && backgroundColor !== 'rgba(0, 0, 0, 0)') {
        // This is a simplified check - in production, you'd want more sophisticated color parsing
        if (color === backgroundColor) {
          issues.push({
            issue: 'Potentially insufficient color contrast',
            element: el,
            severity: 'warning'
          });
        }
      }
    });
    
    return issues;
  }
};

/**
 * Screen reader utilities
 */
export const ScreenReader = {
  /**
   * Check if screen reader is likely active
   */
  isLikelyActive(): boolean {
    // This is a heuristic approach - not 100% reliable
    return (
      navigator.userAgent.includes('NVDA') ||
      navigator.userAgent.includes('JAWS') ||
      window.navigator.userAgent.includes('NARRATOR') ||
      'speechSynthesis' in window
    );
  },
  
  /**
   * Create screen reader only text
   */
  createSROnlyText(text: string): HTMLElement {
    const element = document.createElement('span');
    element.className = 'sr-only';
    element.textContent = text;
    return element;
  }
};