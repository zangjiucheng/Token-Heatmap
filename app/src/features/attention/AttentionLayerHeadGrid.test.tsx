import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AttentionLayerHeadGrid } from './AttentionLayerHeadGrid';
import { makeAttentionTrace } from './testFixtures';

describe('AttentionLayerHeadGrid', () => {
  it('renders one cell per (layer, head)', () => {
    const trace = makeAttentionTrace();
    render(
      <AttentionLayerHeadGrid
        trace={trace}
        selectedStep={0}
        selectedHead={null}
        onSelectHead={() => {}}
      />,
    );
    const numLayers = trace.steps[0].attention!.length;
    const numHeads = trace.attention_metadata!.num_attention_heads;
    const cells = screen.getAllByTestId(/^attention-cell-\d+-\d+$/);
    expect(cells).toHaveLength(numLayers * numHeads);
  });

  it('calls onSelectHead when a cell is clicked', () => {
    const onSelectHead = vi.fn();
    render(
      <AttentionLayerHeadGrid
        trace={makeAttentionTrace()}
        selectedStep={0}
        selectedHead={null}
        onSelectHead={onSelectHead}
      />,
    );
    fireEvent.click(screen.getByTestId('attention-cell-2-1'));
    expect(onSelectHead).toHaveBeenCalledWith(2, 1);
  });

  it('updates cell colors when the metric switches', () => {
    render(
      <AttentionLayerHeadGrid
        trace={makeAttentionTrace()}
        selectedStep={0}
        selectedHead={null}
        onSelectHead={() => {}}
      />,
    );
    const cellEntropy = screen.getByTestId('attention-cell-0-0');
    const before = cellEntropy.getAttribute('fill');
    fireEvent.change(screen.getByTestId('attention-metric-select'), {
      target: { value: 'bos_weight' },
    });
    const after = screen.getByTestId('attention-cell-0-0').getAttribute('fill');
    // Different metric implies different value-range mapping, so the color
    // for the same cell should change.
    expect(after).not.toBe(before);
  });

  it('prompts the user to choose a generation step before rendering attention heads', () => {
    render(
      <AttentionLayerHeadGrid
        trace={makeAttentionTrace()}
        selectedStep={null}
        selectedHead={null}
        onSelectHead={() => {}}
      />,
    );

    expect(screen.getByTestId('attention-layer-head-grid')).toHaveTextContent(
      /select a generation step/i,
    );
    expect(screen.queryByTestId(/^attention-cell-\d+-\d+$/)).toBeNull();
  });
});
