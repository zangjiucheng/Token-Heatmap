import type { ViewerTab } from '@/hooks/useViewState';
import { ChevronIcon, LensIcon } from './icons';
import {
  isLensAvailable,
  LENS_GROUP_LABELS,
  LENS_GROUP_ORDER,
  lensesInGroup,
  type LensAvailability,
} from './lenses';
import './LensRail.css';

export interface LensRailProps {
  activeLens: ViewerTab;
  availability: LensAvailability;
  onSelect: (lens: ViewerTab) => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

/**
 * Vertical, grouped navigation for the analysis lenses — replaces the flat
 * seven-tab row. Locked lenses (missing capture flags) stay visible but
 * disabled with an explanatory tooltip, so the menu also documents what a
 * richer re-run would unlock. Collapses to an icon-only rail.
 *
 * A plain <nav> with `aria-current` is used rather than the ARIA tab pattern
 * because the rail is grouped (a tablist may only contain tabs) and the lens
 * bodies are not wired as labelled tabpanels.
 */
export function LensRail({
  activeLens,
  availability,
  onSelect,
  collapsed,
  onToggleCollapsed,
}: LensRailProps) {
  return (
    <nav
      className="lens-rail"
      aria-label="Analysis lenses"
      data-testid="lens-rail"
      data-collapsed={collapsed ? 'true' : 'false'}
    >
      <button
        type="button"
        className="lens-rail__collapse"
        onClick={onToggleCollapsed}
        aria-expanded={!collapsed}
        aria-label={collapsed ? 'Expand lens rail' : 'Collapse lens rail'}
        title={collapsed ? 'Expand lens rail' : 'Collapse lens rail'}
        data-testid="lens-rail-collapse"
      >
        <ChevronIcon direction={collapsed ? 'right' : 'left'} />
        {!collapsed && <span className="lens-rail__collapse-label">Lenses</span>}
      </button>

      {LENS_GROUP_ORDER.map((group) => (
        <div
          key={group}
          role="group"
          aria-label={LENS_GROUP_LABELS[group]}
          className="lens-rail__group"
        >
          <div className="lens-rail__group-label eyebrow" aria-hidden="true">
            {LENS_GROUP_LABELS[group]}
          </div>
          {lensesInGroup(group).map((lens) => {
            const available = isLensAvailable(lens, availability);
            const active = activeLens === lens.id;
            return (
              <button
                key={lens.id}
                type="button"
                className="lens-rail__item"
                data-testid={lens.testId}
                data-active={active ? 'true' : 'false'}
                aria-current={active ? 'page' : undefined}
                aria-label={lens.label}
                disabled={!available}
                title={available ? lens.label : lens.lockedHint}
                onClick={() => onSelect(lens.id)}
              >
                <span className="lens-rail__item-icon" aria-hidden="true">
                  <LensIcon lens={lens.id} />
                </span>
                <span className="lens-rail__item-label">{lens.label}</span>
                {!available && (
                  <span className="lens-rail__lock" aria-hidden="true">
                    ⤬
                  </span>
                )}
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
}
