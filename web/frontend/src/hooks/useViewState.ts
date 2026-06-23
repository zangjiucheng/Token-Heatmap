import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { ComparisonMode } from '@/features/comparison';
import type { ValueCol } from '@/features/heatmap';
import type { ColorRangeValue } from '@/features/controls';

export type ViewerTab = 'heatmap' | 'attention' | 'logit-lens' | 'activations';

export interface SelectedHead {
  layer: number;
  head: number;
}

export interface ViewState {
  mode: ComparisonMode;
  valueCol: ValueCol;
  /** `null` means "full trace" / no clipping. */
  stepRange: [number, number] | null;
  colorRange: ColorRangeValue;
  /** Whether the left side panel is expanded. Defaults to true. */
  leftOpen: boolean;
  /** Whether the right side panel is expanded. Defaults to true. */
  rightOpen: boolean;
  /** Active top-level tab in the center column. Defaults to 'heatmap'. */
  tab: ViewerTab;
  /** Selected (layer, head) for the Attention tab. `null` when no head is selected. */
  selectedHead: SelectedHead | null;
}

export const DEFAULT_VIEW_STATE: ViewState = {
  mode: 'raw',
  valueCol: 'logprob',
  stepRange: null,
  colorRange: { mode: 'auto', min: null, max: null },
  leftOpen: true,
  rightOpen: true,
  tab: 'heatmap',
  selectedHead: null,
};

const VALID_MODES: ComparisonMode[] = ['raw', 'processed', 'split'];
const VALID_VALUE_COLS: ValueCol[] = ['logprob', 'prob'];
const VALID_TABS: ViewerTab[] = ['heatmap', 'attention', 'logit-lens', 'activations'];

function parseMode(raw: string | null): ComparisonMode {
  if (raw && (VALID_MODES as string[]).includes(raw)) {
    return raw as ComparisonMode;
  }
  return DEFAULT_VIEW_STATE.mode;
}

function parseValueCol(raw: string | null): ValueCol {
  if (raw && (VALID_VALUE_COLS as string[]).includes(raw)) {
    return raw as ValueCol;
  }
  return DEFAULT_VIEW_STATE.valueCol;
}

function parseStepRange(raw: string | null): [number, number] | null {
  if (!raw) return null;
  const parts = raw.split('-');
  if (parts.length !== 2) return null;
  const start = Number(parts[0]);
  const end = Number(parts[1]);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start < 0 || end < 0) return null;
  if (start > end) return null;
  return [Math.floor(start), Math.floor(end)];
}

function parseNumber(raw: string | null): number | null {
  if (raw == null || raw === '') return null;
  const v = Number(raw);
  return Number.isFinite(v) ? v : null;
}

function parsePanelOpen(raw: string | null, defaultOpen: boolean): boolean {
  if (raw == null) return defaultOpen;
  // `0` means collapsed; `1` (or anything else non-empty) means expanded.
  if (raw === '0') return false;
  if (raw === '1') return true;
  return defaultOpen;
}

function parseColorRange(
  rawMin: string | null,
  rawMax: string | null,
): ColorRangeValue {
  const min = parseNumber(rawMin);
  const max = parseNumber(rawMax);
  if (min == null && max == null) return { mode: 'auto', min: null, max: null };
  // If only one bound is provided we still treat the user as having opted
  // into manual mode; the missing bound will be filled by the heatmap's auto
  // value at render time.
  return { mode: 'manual', min, max };
}

function serializeStepRange(range: [number, number] | null): string | null {
  if (!range) return null;
  return `${range[0]}-${range[1]}`;
}

function setOrDelete(
  params: URLSearchParams,
  key: string,
  value: string | null,
): void {
  if (value == null || value === '') {
    params.delete(key);
  } else {
    params.set(key, value);
  }
}

export interface UseViewStateResult {
  state: ViewState;
  setMode: (mode: ComparisonMode) => void;
  setValueCol: (valueCol: ValueCol) => void;
  setStepRange: (range: [number, number] | null) => void;
  setColorRange: (range: ColorRangeValue) => void;
  setLeftOpen: (open: boolean) => void;
  setRightOpen: (open: boolean) => void;
  setTab: (tab: ViewerTab) => void;
  setSelectedHead: (value: SelectedHead | null) => void;
}

function parseTab(raw: string | null): ViewerTab {
  if (raw && (VALID_TABS as string[]).includes(raw)) {
    return raw as ViewerTab;
  }
  return DEFAULT_VIEW_STATE.tab;
}

function parseSelectedHead(raw: string | null): SelectedHead | null {
  if (!raw) return null;
  const parts = raw.split('-');
  if (parts.length !== 2) return null;
  const layer = Number(parts[0]);
  const head = Number(parts[1]);
  if (!Number.isFinite(layer) || !Number.isFinite(head)) return null;
  if (layer < 0 || head < 0) return null;
  return { layer: Math.floor(layer), head: Math.floor(head) };
}

function serializeSelectedHead(value: SelectedHead | null): string | null {
  if (!value) return null;
  return `${value.layer}-${value.head}`;
}

/**
 * Round-trips the comparison-view state to and from the URL search params so
 * the view is shareable and survives a page reload.
 */
export function useViewState(): UseViewStateResult {
  const [searchParams, setSearchParams] = useSearchParams();

  const state = useMemo<ViewState>(() => {
    const mode = parseMode(searchParams.get('mode'));
    const valueCol = parseValueCol(searchParams.get('valueCol'));
    const stepRange = parseStepRange(searchParams.get('stepRange'));
    const colorRange = parseColorRange(
      searchParams.get('colorMin'),
      searchParams.get('colorMax'),
    );
    const leftOpen = parsePanelOpen(
      searchParams.get('left'),
      DEFAULT_VIEW_STATE.leftOpen,
    );
    const rightOpen = parsePanelOpen(
      searchParams.get('right'),
      DEFAULT_VIEW_STATE.rightOpen,
    );
    const tab = parseTab(searchParams.get('tab'));
    const selectedHead = parseSelectedHead(searchParams.get('selectedHead'));
    return {
      mode,
      valueCol,
      stepRange,
      colorRange,
      leftOpen,
      rightOpen,
      tab,
      selectedHead,
    };
  }, [searchParams]);

  const update = useCallback(
    (mutator: (params: URLSearchParams) => void) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          mutator(next);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setMode = useCallback(
    (mode: ComparisonMode) => {
      update((params) => {
        if (mode === DEFAULT_VIEW_STATE.mode) {
          params.delete('mode');
        } else {
          params.set('mode', mode);
        }
      });
    },
    [update],
  );

  const setValueCol = useCallback(
    (valueCol: ValueCol) => {
      update((params) => {
        if (valueCol === DEFAULT_VIEW_STATE.valueCol) {
          params.delete('valueCol');
        } else {
          params.set('valueCol', valueCol);
        }
      });
    },
    [update],
  );

  const setStepRange = useCallback(
    (range: [number, number] | null) => {
      update((params) => {
        setOrDelete(params, 'stepRange', serializeStepRange(range));
      });
    },
    [update],
  );

  const setColorRange = useCallback(
    (range: ColorRangeValue) => {
      update((params) => {
        if (range.mode === 'auto') {
          params.delete('colorMin');
          params.delete('colorMax');
          return;
        }
        setOrDelete(
          params,
          'colorMin',
          range.min == null ? null : String(range.min),
        );
        setOrDelete(
          params,
          'colorMax',
          range.max == null ? null : String(range.max),
        );
      });
    },
    [update],
  );

  const setLeftOpen = useCallback(
    (open: boolean) => {
      update((params) => {
        if (open === DEFAULT_VIEW_STATE.leftOpen) {
          params.delete('left');
        } else {
          params.set('left', open ? '1' : '0');
        }
      });
    },
    [update],
  );

  const setRightOpen = useCallback(
    (open: boolean) => {
      update((params) => {
        if (open === DEFAULT_VIEW_STATE.rightOpen) {
          params.delete('right');
        } else {
          params.set('right', open ? '1' : '0');
        }
      });
    },
    [update],
  );

  const setTab = useCallback(
    (tab: ViewerTab) => {
      update((params) => {
        if (tab === DEFAULT_VIEW_STATE.tab) {
          params.delete('tab');
        } else {
          params.set('tab', tab);
        }
      });
    },
    [update],
  );

  const setSelectedHead = useCallback(
    (value: SelectedHead | null) => {
      update((params) => {
        setOrDelete(params, 'selectedHead', serializeSelectedHead(value));
      });
    },
    [update],
  );

  return {
    state,
    setMode,
    setValueCol,
    setStepRange,
    setColorRange,
    setLeftOpen,
    setRightOpen,
    setTab,
    setSelectedHead,
  };
}
