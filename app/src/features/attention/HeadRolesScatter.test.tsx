import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { Trace } from '@/types/trace';
import { HeadRolesScatter } from './HeadRolesScatter';

function head(
  layer: number,
  h: number,
  role: string,
  dla: number,
  bos: number,
) {
  return {
    layer,
    head: h,
    role,
    dla_mean: dla,
    dla_absmean: Math.abs(dla),
    bos_weight: bos,
    self_weight: 0.1,
    induction: 0.05,
  };
}

function makeTrace(): Trace {
  return {
    head_roles: {
      heads: [
        head(20, 3, 'worker', 0.6, 0.05),
        head(0, 1, 'sink', 0.0, 0.9),
        head(5, 2, 'sink', 0.01, 0.8),
        head(12, 7, 'local', 0.1, 0.2),
      ],
      summary: { counts: { worker: 1, sink: 2, local: 1 } },
    },
  } as unknown as Trace;
}

describe('HeadRolesScatter', () => {
  it('renders a point per head and a role legend with counts', () => {
    render(<HeadRolesScatter trace={makeTrace()} />);
    expect(screen.getByTestId('head-roles-scatter')).toBeInTheDocument();
    expect(screen.getByTestId('head-role-point-20-3')).toBeInTheDocument();
    expect(screen.getByTestId('head-role-point-0-1')).toBeInTheDocument();
    // legend shows the two sinks
    expect(screen.getByText('sink')).toBeInTheDocument();
  });

  it('switches the attention (x) axis', () => {
    render(<HeadRolesScatter trace={makeTrace()} />);
    const self = screen.getByTestId('head-roles-x-self_weight');
    expect(self).toHaveAttribute('data-selected', 'false');
    fireEvent.click(self);
    expect(self).toHaveAttribute('data-selected', 'true');
    expect(screen.getByTestId('head-roles-x-bos_weight')).toHaveAttribute(
      'data-selected',
      'false',
    );
  });

  it('renders nothing without head_roles data', () => {
    const { container } = render(
      <HeadRolesScatter trace={{ steps: [] } as unknown as Trace} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
