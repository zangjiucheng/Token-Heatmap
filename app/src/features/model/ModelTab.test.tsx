import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ModelTab } from './ModelTab';
import {
  makeTraceArchitectureBare,
  makeTraceArchitectureFallback,
  makeTraceWithArchitecture,
} from './testFixtures';

describe('ModelTab', () => {
  it('renders the headline spec card from model_architecture', () => {
    render(<ModelTab trace={makeTraceWithArchitecture()} />);
    expect(screen.getByTestId('model-tab')).toBeInTheDocument();
    expect(
      screen.getByText('Qwen/Qwen2.5-7B-Instruct'),
    ).toBeInTheDocument();
    expect(screen.getByText('Qwen2ForCausalLM')).toBeInTheDocument();
    // Humanized parameter count appears as a hero tag.
    expect(screen.getByText('7.6B params')).toBeInTheDocument();
  });

  it('shows the decoder repeat badge and GQA group', () => {
    render(<ModelTab trace={makeTraceWithArchitecture()} />);
    expect(screen.getByText('× 28')).toBeInTheDocument();
    // 28 query heads / 4 KV heads → GQA ×7.
    expect(screen.getByText(/GQA ×7/)).toBeInTheDocument();
  });

  it('lists key dimensions in the specs table', () => {
    render(<ModelTab trace={makeTraceWithArchitecture()} />);
    const specs = screen.getByLabelText('Model dimensions');
    expect(within(specs).getByText('Vocab size')).toBeInTheDocument();
    expect(within(specs).getByText('152,064')).toBeInTheDocument();
    expect(within(specs).getByText('MLP intermediate')).toBeInTheDocument();
    expect(within(specs).getByText('18,944')).toBeInTheDocument();
  });

  it('falls back to attention metadata when model_architecture is absent', () => {
    render(<ModelTab trace={makeTraceArchitectureFallback()} />);
    // Layer count and head count still render from attention_metadata.
    expect(screen.getByText('× 28')).toBeInTheDocument();
    expect(screen.getByText(/attention · 3 layers/)).toBeInTheDocument();
  });

  it('shows the empty state when no dimensions are available', () => {
    render(<ModelTab trace={makeTraceArchitectureBare()} />);
    expect(screen.getByTestId('model-tab-empty')).toBeInTheDocument();
  });
});
