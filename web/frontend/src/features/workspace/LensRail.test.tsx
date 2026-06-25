import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { LensRail } from './LensRail';
import type { LensAvailability } from './lenses';

const ALL_AVAILABLE: LensAvailability = {
  attention: true,
  logitLens: true,
  activations: true,
  manifold: true,
};

const NONE_AVAILABLE: LensAvailability = {
  attention: false,
  logitLens: false,
  activations: false,
  manifold: false,
};

function renderRail(overrides: Partial<Parameters<typeof LensRail>[0]> = {}) {
  const onSelect = vi.fn();
  const onToggleCollapsed = vi.fn();
  render(
    <LensRail
      activeLens="heatmap"
      availability={ALL_AVAILABLE}
      onSelect={onSelect}
      collapsed={false}
      onToggleCollapsed={onToggleCollapsed}
      {...overrides}
    />,
  );
  return { onSelect, onToggleCollapsed };
}

describe('LensRail', () => {
  it('groups the lenses under Generation / Internals / Geometry', () => {
    renderRail();
    const nav = screen.getByRole('navigation', { name: /analysis lenses/i });
    expect(nav).toBeInTheDocument();
    expect(screen.getByRole('group', { name: /generation/i })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: /internals/i })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: /geometry/i })).toBeInTheDocument();
  });

  it('marks the active lens with aria-current', () => {
    renderRail({ activeLens: 'attention' });
    expect(screen.getByTestId('attention-tab')).toHaveAttribute(
      'aria-current',
      'page',
    );
    expect(screen.getByTestId('heatmap-tab')).not.toHaveAttribute(
      'aria-current',
    );
  });

  it('selects a lens on click', async () => {
    const { onSelect } = renderRail();
    await userEvent.click(screen.getByTestId('model-tab'));
    expect(onSelect).toHaveBeenCalledWith('model');
  });

  it('disables lenses whose capture flag is missing, with an explanatory tooltip', () => {
    renderRail({ availability: NONE_AVAILABLE });
    const attention = screen.getByTestId('attention-tab');
    expect(attention).toBeDisabled();
    expect(attention).toHaveAttribute('title', expect.stringMatching(/--capture-attention/));
    // Generation lenses never gate on a capture flag.
    expect(screen.getByTestId('heatmap-tab')).toBeEnabled();
  });

  it('toggles the collapsed state', async () => {
    const { onToggleCollapsed } = renderRail({ collapsed: true });
    const nav = screen.getByTestId('lens-rail');
    expect(nav).toHaveAttribute('data-collapsed', 'true');
    await userEvent.click(
      screen.getByRole('button', { name: /expand lens rail/i }),
    );
    expect(onToggleCollapsed).toHaveBeenCalledTimes(1);
  });
});
