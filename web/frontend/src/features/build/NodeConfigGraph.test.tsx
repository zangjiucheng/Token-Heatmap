import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { NodeConfigGraph } from './NodeConfigGraph';
import { DEFAULT_BUILD_CONFIG, type BuildConfig } from './config';

function setup(overrides: Partial<BuildConfig> = {}, props = {}) {
  const onChange = vi.fn();
  const onRun = vi.fn();
  const onExport = vi.fn();
  const config = { ...DEFAULT_BUILD_CONFIG, ...overrides };
  render(
    <NodeConfigGraph
      config={config}
      onChange={onChange}
      onRun={onRun}
      onExport={onExport}
      {...props}
    />,
  );
  return { onChange, onRun, onExport };
}

describe('NodeConfigGraph', () => {
  it('renders all five pipeline nodes', () => {
    setup();
    for (let step = 1; step <= 5; step += 1) {
      expect(screen.getByTestId(`build-node-${step}`)).toBeInTheDocument();
    }
  });

  it('disables Run and Export until a prompt is set', () => {
    setup({ prompt: '' });
    expect(screen.getByTestId('node-run')).toBeDisabled();
    expect(screen.getByTestId('node-export')).toBeDisabled();
    expect(screen.getByTestId('node-invalid-hint')).toBeInTheDocument();
  });

  it('enables and fires Run / Export when valid', async () => {
    const user = userEvent.setup();
    const { onRun, onExport } = setup({ prompt: 'hi there' });
    await user.click(screen.getByTestId('node-run'));
    await user.click(screen.getByTestId('node-export'));
    expect(onRun).toHaveBeenCalledTimes(1);
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it('emits patches as fields change', async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ prompt: 'hi' });
    await user.click(screen.getByTestId('node-capture-attention'));
    expect(onChange).toHaveBeenCalledWith({ capture_attention: true });
  });

  it('shows a running state and a backend error', () => {
    setup({ prompt: 'hi' }, { running: true, error: 'boom' });
    expect(screen.getByTestId('node-run')).toHaveTextContent('Running');
    expect(screen.getByTestId('node-run')).toBeDisabled();
    expect(screen.getByTestId('node-error')).toHaveTextContent('boom');
  });

  it('selects a model preset from the searchable picker', async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ prompt: 'hi' });
    // Open the combobox, then pick a preset from the dropdown.
    await user.click(screen.getByTestId('node-input-model'));
    await user.click(screen.getByText('Qwen2.5-7B-Instruct'));
    expect(onChange).toHaveBeenCalledWith({
      model: 'Qwen/Qwen2.5-7B-Instruct',
    });
  });
});
