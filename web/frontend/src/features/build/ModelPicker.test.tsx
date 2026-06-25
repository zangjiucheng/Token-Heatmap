import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ModelPicker } from './ModelPicker';

function setup(value = 'Qwen/Qwen2.5-0.5B-Instruct') {
  const onChange = vi.fn();
  render(<ModelPicker value={value} onChange={onChange} />);
  return { onChange };
}

describe('ModelPicker', () => {
  it('opens the grouped preset list on focus', async () => {
    const user = userEvent.setup();
    setup();
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    await user.click(screen.getByRole('combobox', { name: /model/i }));
    expect(screen.getByRole('listbox')).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Qwen' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Llama' })).toBeInTheDocument();
  });

  it('selects a preset and closes', async () => {
    const user = userEvent.setup();
    const { onChange } = setup();
    await user.click(screen.getByRole('combobox', { name: /model/i }));
    await user.click(screen.getByText('Qwen2.5-14B-Instruct'));
    expect(onChange).toHaveBeenCalledWith('Qwen/Qwen2.5-14B-Instruct');
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('filters presets by a typed query', async () => {
    const user = userEvent.setup();
    // A non-preset value acts as the filter query.
    setup('llama');
    await user.click(screen.getByRole('combobox', { name: /model/i }));
    expect(screen.getByText('Llama-3.1-8B-Instruct')).toBeInTheDocument();
    expect(screen.queryByText('Qwen2.5-7B-Instruct')).not.toBeInTheDocument();
  });

  it('treats an unmatched value as a custom model id', async () => {
    const user = userEvent.setup();
    setup('bigscience/bloom-560m');
    await user.click(screen.getByRole('combobox', { name: /model/i }));
    expect(screen.getByText(/custom model id/i)).toBeInTheDocument();
  });

  it('shows a size badge for the current model', () => {
    setup('Qwen/Qwen2.5-32B-Instruct');
    expect(screen.getByText('32B')).toBeInTheDocument();
  });
});
