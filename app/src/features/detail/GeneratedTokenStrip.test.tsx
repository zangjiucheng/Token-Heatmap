import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { Trace, Step } from '@/types/trace';
import { GeneratedTokenStrip } from './GeneratedTokenStrip';

function makeStep(idx: number, token: string): Step {
  return {
    step: idx,
    selected: { token_id: 100 + idx, token },
    raw: {
      k_used: 1,
      entropy: 0,
      top_mass_used: 1,
      selected_prob: 1,
      selected_logprob: 0,
      selected_rank: 1,
      candidates: [
        { rank: 1, token_id: 100 + idx, token, prob: 1, logprob: 0 },
      ],
    },
    processed: {
      k_used: 1,
      entropy: 0,
      top_mass_used: 1,
      selected_prob: 1,
      selected_logprob: 0,
      selected_rank: 1,
      candidates: [
        { rank: 1, token_id: 100 + idx, token, prob: 1, logprob: 0 },
      ],
    },
  };
}

function makeTrace(tokens: string[]): Trace {
  return {
    schema_version: '2.0.0',
    metadata: {
      model: 'fake',
      prompt: '',
      generated_text: tokens.join(''),
      generated_at: '2026-05-13T00:00:00Z',
      generation_params: {
        max_new_tokens: tokens.length,
        temperature: 1,
        top_p: 1,
        sample_top_k: 0,
      },
      probe_config: { min_k: 1, max_k: 1, mass_threshold: 0.9 },
    },
    tokens: { prompt_token_ids: [], prompt_tokens: [] },
    steps: tokens.map((t, i) => makeStep(i, t)),
  };
}

describe('GeneratedTokenStrip', () => {
  it('renders one button per generation step with escaped token text', () => {
    const onSelect = vi.fn();
    render(
      <GeneratedTokenStrip
        trace={makeTrace(['Hello', ' world', '\n', '!'])}
        selectedStep={null}
        onSelectStep={onSelect}
      />,
    );
    expect(screen.getByTestId('generated-token-0')).toHaveTextContent('Hello');
    expect(screen.getByTestId('generated-token-1').textContent).toBe(' world');
    // \n should be rendered as the escaped literal "\n".
    expect(screen.getByTestId('generated-token-2')).toHaveTextContent('\\n');
    expect(screen.getByTestId('generated-token-3')).toHaveTextContent('!');
  });

  it('fires onSelectStep with the step index when a token is clicked', () => {
    const onSelect = vi.fn();
    render(
      <GeneratedTokenStrip
        trace={makeTrace(['a', 'b', 'c'])}
        selectedStep={null}
        onSelectStep={onSelect}
      />,
    );
    fireEvent.click(screen.getByTestId('generated-token-1'));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it('marks the selected step with the selected modifier and aria-pressed', () => {
    render(
      <GeneratedTokenStrip
        trace={makeTrace(['a', 'b', 'c'])}
        selectedStep={1}
        onSelectStep={() => {}}
      />,
    );
    const selected = screen.getByTestId('generated-token-1');
    expect(
      selected.classList.contains('generated-token-strip__token--selected'),
    ).toBe(true);
    expect(selected.getAttribute('aria-pressed')).toBe('true');
    const other = screen.getByTestId('generated-token-0');
    expect(
      other.classList.contains('generated-token-strip__token--selected'),
    ).toBe(false);
    expect(other.getAttribute('aria-pressed')).toBe('false');
  });

  it('forwards hover events when an onHoverStep handler is supplied', () => {
    const onHover = vi.fn();
    render(
      <GeneratedTokenStrip
        trace={makeTrace(['a', 'b'])}
        selectedStep={null}
        onSelectStep={() => {}}
        onHoverStep={onHover}
      />,
    );
    fireEvent.mouseEnter(screen.getByTestId('generated-token-1'));
    expect(onHover).toHaveBeenLastCalledWith(1);
    fireEvent.mouseLeave(screen.getByTestId('generated-token-1'));
    expect(onHover).toHaveBeenLastCalledWith(null);
  });

  it('renders nothing when the trace has no steps', () => {
    render(
      <GeneratedTokenStrip
        trace={makeTrace([])}
        selectedStep={null}
        onSelectStep={() => {}}
      />,
    );
    expect(screen.queryByTestId('generated-token-strip')).toBeNull();
  });
});
