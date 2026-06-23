import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { GenerateForm } from './GenerateForm';

describe('GenerateForm', () => {
  it('disables submit until a prompt is entered', async () => {
    const onGenerate = vi.fn();
    render(<GenerateForm onGenerate={onGenerate} />);
    const submit = screen.getByRole('button', { name: /generate/i });
    // Model has a default, prompt does not → submit starts disabled.
    expect(submit).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/prompt/i), 'hello');
    expect(submit).toBeEnabled();
  });

  it('submits the default params with numeric types parsed', async () => {
    const onGenerate = vi.fn();
    render(<GenerateForm onGenerate={onGenerate} />);
    await userEvent.type(screen.getByLabelText(/prompt/i), 'hi');
    await userEvent.click(screen.getByRole('button', { name: /generate/i }));

    expect(onGenerate).toHaveBeenCalledTimes(1);
    const params = onGenerate.mock.calls[0][0];
    expect(params).toMatchObject({
      model: 'Qwen/Qwen2.5-0.5B-Instruct',
      prompt: 'hi',
      max_new_tokens: 64,
      temperature: 0.8,
      top_p: 0.95,
      min_k: 8,
      max_k: 64,
      mass_threshold: 0.95,
      capture_attention: false,
      capture_logit_lens: false,
      capture_activations: false,
    });
    expect(typeof params.max_new_tokens).toBe('number');
    expect(typeof params.temperature).toBe('number');
  });

  it('reflects capture toggles in the submitted params', async () => {
    const onGenerate = vi.fn();
    render(<GenerateForm onGenerate={onGenerate} />);
    await userEvent.type(screen.getByLabelText(/prompt/i), 'hi');
    await userEvent.click(screen.getByText('Advanced'));
    await userEvent.click(screen.getByLabelText(/attention/i));
    await userEvent.click(screen.getByRole('button', { name: /generate/i }));

    expect(onGenerate.mock.calls[0][0].capture_attention).toBe(true);
    expect(onGenerate.mock.calls[0][0].capture_logit_lens).toBe(false);
  });

  it('does not submit when disabled', async () => {
    const onGenerate = vi.fn();
    render(<GenerateForm onGenerate={onGenerate} disabled />);
    const submit = screen.getByRole('button', { name: /generate/i });
    expect(submit).toBeDisabled();
    await userEvent.click(submit);
    expect(onGenerate).not.toHaveBeenCalled();
  });
});
