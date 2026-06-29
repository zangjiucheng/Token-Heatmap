import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import axe from 'axe-core';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { TokenHeatmap } from './TokenHeatmap';

const trace = sampleTrace as unknown as Trace;

describe('TokenHeatmap accessibility', () => {
  it('has zero axe-core violations on a standalone render', async () => {
    const { container } = render(
      <TokenHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={800}
        height={400}
      />,
    );
    const results = await axe.run(container, {
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    expect(results.violations).toEqual([]);
  });
});
