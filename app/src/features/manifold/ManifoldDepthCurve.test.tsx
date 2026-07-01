import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ManifoldLayer } from '@/types/trace';
import { ManifoldDepthCurve } from './ManifoldDepthCurve';

function layer(
  l: number,
  submodule: string,
  twonn: number,
  pr: number,
): ManifoldLayer {
  return {
    layer: l,
    submodule,
    intrinsic_dimension: { twonn },
    participation_ratio: pr,
  } as unknown as ManifoldLayer;
}

function makeLayers(): ManifoldLayer[] {
  const out: ManifoldLayer[] = [];
  for (const sub of ['mlp_out', 'resid_post']) {
    for (let l = 0; l < 4; l += 1) out.push(layer(l, sub, 3 + l, 2 + l));
  }
  return out;
}

describe('ManifoldDepthCurve', () => {
  it('draws one line per submodule with a legend', () => {
    render(<ManifoldDepthCurve layers={makeLayers()} />);
    expect(screen.getByTestId('manifold-depth-curve')).toBeInTheDocument();
    expect(screen.getByTestId('depth-curve-line-mlp_out')).toBeInTheDocument();
    expect(
      screen.getByTestId('depth-curve-line-resid_post'),
    ).toBeInTheDocument();
  });

  it('toggles between intrinsic-dim and participation metrics', () => {
    render(<ManifoldDepthCurve layers={makeLayers()} />);
    const pr = screen.getByTestId('depth-curve-metric-pr');
    expect(pr).toHaveAttribute('data-selected', 'false');
    fireEvent.click(pr);
    expect(pr).toHaveAttribute('data-selected', 'true');
    expect(screen.getByTestId('depth-curve-metric-id')).toHaveAttribute(
      'data-selected',
      'false',
    );
  });

  it('renders nothing with fewer than two distinct layers', () => {
    const { container } = render(
      <ManifoldDepthCurve layers={[layer(0, 'mlp_out', 3, 2)]} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
