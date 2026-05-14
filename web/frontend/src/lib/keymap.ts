/**
 * Single source of truth for application-wide keybindings.
 *
 * Each binding has a key (or key sequence), a category for grouping in the
 * help dialog, a human-readable description, and an `id` that handlers
 * register against via `useKeymap`. Sequences like `g d` are expressed as
 * two-element tuples; everything else is a single key with optional
 * modifiers expressed as a Combo descriptor.
 */

export type Modifier = 'ctrl' | 'meta' | 'alt' | 'shift';

export interface KeyCombo {
  /** Lowercase `event.key` value (e.g. `'?'`, `'arrowleft'`, `'home'`). */
  key: string;
  modifiers?: ReadonlyArray<Modifier>;
}

export type KeyTrigger = KeyCombo | readonly [KeyCombo, KeyCombo];

export type KeymapCategory =
  | 'navigation'
  | 'selection'
  | 'view'
  | 'comparison'
  | 'layout'
  | 'help';

export interface KeymapBinding {
  id: string;
  trigger: KeyTrigger;
  category: KeymapCategory;
  description: string;
  /** Display string shown in the help dialog (e.g. `'←'`, `'g d'`). */
  display: string;
}

/**
 * Canonical list of supported shortcuts. Keep alphabetised within a category
 * so the help dialog reads predictably.
 */
export const KEYMAP: ReadonlyArray<KeymapBinding> = [
  // Selection
  {
    id: 'selection.prev',
    trigger: { key: 'arrowleft' },
    category: 'selection',
    description: 'Move selection to the previous step',
    display: '←',
  },
  {
    id: 'selection.next',
    trigger: { key: 'arrowright' },
    category: 'selection',
    description: 'Move selection to the next step',
    display: '→',
  },
  {
    id: 'selection.first',
    trigger: { key: 'home' },
    category: 'selection',
    description: 'Jump to the first step',
    display: 'Home',
  },
  {
    id: 'selection.last',
    trigger: { key: 'end' },
    category: 'selection',
    description: 'Jump to the last step',
    display: 'End',
  },
  {
    id: 'selection.clear',
    trigger: { key: 'escape' },
    category: 'selection',
    description: 'Clear tooltip / selection',
    display: 'Esc',
  },

  // View
  {
    id: 'view.toggleTheme',
    trigger: { key: 't' },
    category: 'view',
    description: 'Toggle light/dark theme',
    display: 'T',
  },
  {
    id: 'view.reset',
    trigger: { key: 'r' },
    category: 'view',
    description: 'Reset zoom and pan',
    display: 'R',
  },

  // Comparison
  {
    id: 'comparison.cycle',
    trigger: { key: 'c' },
    category: 'comparison',
    description: 'Cycle distribution mode (raw → processed → split)',
    display: 'C',
  },

  // Layout
  {
    id: 'panel.toggleLeft',
    trigger: { key: '[' },
    category: 'layout',
    description: 'Toggle the left side panel (Controls)',
    display: '[',
  },
  {
    id: 'panel.toggleRight',
    trigger: { key: ']' },
    category: 'layout',
    description: 'Toggle the right side panel (Step detail)',
    display: ']',
  },

  // Navigation
  {
    id: 'navigation.gotoDetail',
    trigger: [{ key: 'g' }, { key: 'd' }],
    category: 'navigation',
    description: 'Focus the step detail panel',
    display: 'G then D',
  },
  {
    id: 'navigation.gotoHeatmap',
    trigger: [{ key: 'g' }, { key: 'h' }],
    category: 'navigation',
    description: 'Switch to the Token Heatmap tab and focus the heatmap',
    display: 'G then H',
  },
  {
    id: 'navigation.gotoAttention',
    trigger: [{ key: 'g' }, { key: 'a' }],
    category: 'navigation',
    description: 'Switch to the Attention tab',
    display: 'G then A',
  },

  // Help
  {
    id: 'help.open',
    trigger: { key: '?' },
    category: 'help',
    description: 'Open the keyboard shortcut help dialog',
    display: '?',
  },
];

export const CATEGORY_LABELS: Record<KeymapCategory, string> = {
  navigation: 'Navigation',
  selection: 'Selection',
  view: 'View',
  comparison: 'Comparison',
  layout: 'Layout',
  help: 'Help',
};

export function getBinding(id: string): KeymapBinding | undefined {
  return KEYMAP.find((b) => b.id === id);
}

function comboMatches(combo: KeyCombo, event: KeyboardEvent): boolean {
  const eventKey = event.key.toLowerCase();
  if (eventKey !== combo.key.toLowerCase()) return false;
  const required = new Set(combo.modifiers ?? []);
  const has = (mod: Modifier) => required.has(mod);
  if (event.ctrlKey !== has('ctrl')) return false;
  if (event.metaKey !== has('meta')) return false;
  if (event.altKey !== has('alt')) return false;
  // Shift is ignored for printable keys whose own keyboard layout already
  // encodes the shift state (e.g. `?` requires Shift on US layouts). Only
  // enforce shift when explicitly declared.
  if (has('shift') && !event.shiftKey) return false;
  return true;
}

export function matchesSingle(combo: KeyCombo, event: KeyboardEvent): boolean {
  return comboMatches(combo, event);
}

/**
 * Whether the current focus is inside an editable element where keyboard
 * shortcuts should be suppressed so the user can type freely.
 */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (target.isContentEditable) return true;
  return false;
}
