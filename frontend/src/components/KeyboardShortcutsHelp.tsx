import React from 'react';
import { commonShortcuts } from '../lib/keyboardShortcuts';

export function KeyboardShortcutsHelp() {
  const shortcuts = Object.entries(commonShortcuts).map(([key, shortcut]) => ({
    ...shortcut,
    id: key
  }));

  const formatKeyCombo = (shortcut: any) => {
    const parts = [];
    if (shortcut.ctrl) parts.push('Ctrl');
    if (shortcut.shift) parts.push('Shift');
    if (shortcut.alt) parts.push('Alt');
    parts.push(shortcut.key.toUpperCase());
    return parts.join(' + ');
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-md">
      <h3 className="text-lg font-semibold mb-4">Keyboard Shortcuts</h3>
      <div className="space-y-3">
        {shortcuts.map((shortcut: any) => (
          <div key={shortcut.id} className="flex justify-between items-center">
            <span className="text-sm text-gray-600">{shortcut.description}</span>
            <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-300 rounded">
              {formatKeyCombo(shortcut)}
            </kbd>
          </div>
        ))}
      </div>
    </div>
  );
} 