import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LogitLensKlChart, type KlLensLayer } from './LogitLensKlChart';

function makeLayers(): KlLensLayer[] {
  return [
    {
      layer_idx: 0,
      kl_to_final: 3.2,
      top_k: [{ token_id: 9, token: ' the', prob: 0.2 }],
    },
    {
      layer_idx: 1,
      kl_to_final: 1.4,
      kl_from_prev: 2.1,
      top_k: [{ token_id: 9, token: ' the', prob: 0.35 }],
    },
    {
      layer_idx: 2,
      kl_to_final: 0.0,
      kl_from_prev: 1.6,
      top_k: [{ token_id: 7, token: ' world', prob: 0.6 }],
    },
  ];
}

describe('LogitLensKlChart', () => {
  it('renders both KL series and the legend', () => {
    render(<LogitLensKlChart layers={makeLayers()} caption="step 0" />);

    expect(screen.getByTestId('logit-lens-kl-chart')).toBeInTheDocument();
    expect(screen.getByTestId('kl-to-final-path')).toBeInTheDocument();
    expect(screen.getByTestId('kl-from-prev-path')).toBeInTheDocument();
    expect(screen.getByText(/Prediction movement through depth — step 0/)).toBeInTheDocument();
    expect(screen.getByText('KL(final ‖ layer)')).toBeInTheDocument();
    expect(screen.getByText('KL(layer ‖ previous)')).toBeInTheDocument();
  });

  it('marks the layers where the top-1 token flipped identity', () => {
    render(<LogitLensKlChart layers={makeLayers()} />);

    // Only layer 2 changes top-1 (9 -> 7); layer 1 keeps token 9.
    expect(screen.getByTestId('kl-flip-marker-2')).toBeInTheDocument();
    expect(screen.queryByTestId('kl-flip-marker-1')).not.toBeInTheDocument();
  });

  it('renders nothing for traces without KL metrics', () => {
    const layers = makeLayers().map(
      ({ layer_idx, top_k }): KlLensLayer => ({ layer_idx, top_k }),
    );
    const { container } = render(<LogitLensKlChart layers={layers} />);
    expect(container).toBeEmptyDOMElement();
  });
});
