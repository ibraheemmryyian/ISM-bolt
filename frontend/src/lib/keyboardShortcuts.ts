import { useEffect } from 'react';

interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  description: string;
  action: () => void;
}

interface KeyboardShortcutsConfig {
  shortcuts: KeyboardShortcut[];
  enabled?: boolean;
}

export function useKeyboardShortcuts({ shortcuts, enabled = true }: KeyboardShortcutsConfig) {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in input fields
      if (event.target instanceof HTMLInputElement || 
          event.target instanceof HTMLTextAreaElement ||
          event.target instanceof HTMLSelectElement) {
        return;
      }

      shortcuts.forEach(shortcut => {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          shortcut.action();
        }
      });
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled]);
}

// Common keyboard shortcuts for the app
export const commonShortcuts = {
  // Navigation
  goToDashboard: {
    key: 'd',
    ctrl: true,
    description: 'Go to Dashboard',
    action: () => window.location.href = '/dashboard'
  },
  
  goToMarketplace: {
    key: 'm',
    ctrl: true,
    description: 'Go to Marketplace',
    action: () => window.location.href = '/marketplace'
  },
  
  goToHome: {
    key: 'h',
    ctrl: true,
    description: 'Go to Home',
    action: () => window.location.href = '/'
  },
  
  // Actions
  newMaterial: {
    key: 'n',
    ctrl: true,
    description: 'New Material Listing',
    action: () => {
      // Trigger new material form
      const event = new CustomEvent('openMaterialForm', { detail: 'waste' });
      window.dispatchEvent(event);
    }
  },
  
  search: {
    key: 'f',
    ctrl: true,
    description: 'Search',
    action: () => {
      const searchInput = document.querySelector('[data-search-input]') as HTMLInputElement;
      if (searchInput) {
        searchInput.focus();
      }
    }
  },
  
  // System
  signOut: {
    key: 'q',
    ctrl: true,
    shift: true,
    description: 'Sign Out',
    action: () => {
      // Trigger sign out
      const event = new CustomEvent('signOut');
      window.dispatchEvent(event);
    }
  },
  
  help: {
    key: '?',
    description: 'Show Help',
    action: () => {
      // Show help modal or documentation
      const event = new CustomEvent('showHelp');
      window.dispatchEvent(event);
    }
  },
  
  // Admin
  adminPanel: {
    key: 'a',
    ctrl: true,
    shift: true,
    description: 'Admin Panel',
    action: () => window.location.href = '/admin'
  }
};

// Hook for using common shortcuts
export function useCommonShortcuts() {
  const shortcuts = Object.values(commonShortcuts).map(shortcut => ({
    key: shortcut.key,
    ctrl: shortcut.ctrl,
    shift: shortcut.shift,
    alt: shortcut.alt,
    description: shortcut.description,
    action: shortcut.action
  }));

  useKeyboardShortcuts({ shortcuts });
}

 