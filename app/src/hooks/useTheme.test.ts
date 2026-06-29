import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useTheme } from '@/hooks/useTheme';

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
});

describe('useTheme', () => {
  it('defaults to dark when no override is stored', () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('reads the stored override on mount in preference to the dark default', () => {
    window.localStorage.setItem('llm-heatmap-theme', 'light');
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('persists the user override to localStorage', () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('light'));
    expect(result.current.theme).toBe('light');
    expect(window.localStorage.getItem('llm-heatmap-theme')).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('toggleTheme flips and persists the value', () => {
    const { result } = renderHook(() => useTheme());
    // Starts from the dark default.
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('light');
    expect(window.localStorage.getItem('llm-heatmap-theme')).toBe('light');
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('dark');
    expect(window.localStorage.getItem('llm-heatmap-theme')).toBe('dark');
  });
});
