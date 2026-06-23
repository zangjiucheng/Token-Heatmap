import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { EmptyState } from '@/components/feedback/EmptyState';

describe('EmptyState', () => {
  it('invokes onLoadSample when the sample button is clicked', async () => {
    const onLoadSample = vi.fn();
    render(<EmptyState onLoadSample={onLoadSample} />);
    await userEvent.click(
      screen.getByRole('button', { name: /try sample data/i }),
    );
    expect(onLoadSample).toHaveBeenCalledTimes(1);
  });

  it('renders the heading and description', () => {
    render(<EmptyState onLoadSample={() => undefined} />);
    expect(
      screen.getByRole('heading', { name: /no trace loaded/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/drop a json or csv trace file/i),
    ).toBeInTheDocument();
  });

  it('omits the URL field when onUrlSubmit is not provided', () => {
    render(<EmptyState onLoadSample={() => undefined} />);
    expect(screen.queryByLabelText(/trace url/i)).not.toBeInTheDocument();
  });

  it('submits the trimmed URL via onUrlSubmit', async () => {
    const onUrlSubmit = vi.fn();
    render(
      <EmptyState onLoadSample={() => undefined} onUrlSubmit={onUrlSubmit} />,
    );

    const input = screen.getByLabelText(/trace url/i);
    await userEvent.type(input, '  http://localhost:8000/trace.json  ');
    await userEvent.click(screen.getByRole('button', { name: /^load$/i }));

    expect(onUrlSubmit).toHaveBeenCalledTimes(1);
    expect(onUrlSubmit).toHaveBeenCalledWith(
      'http://localhost:8000/trace.json',
    );
  });

  it('disables Load until a URL is entered', async () => {
    const onUrlSubmit = vi.fn();
    render(
      <EmptyState onLoadSample={() => undefined} onUrlSubmit={onUrlSubmit} />,
    );

    const loadButton = screen.getByRole('button', { name: /^load$/i });
    expect(loadButton).toBeDisabled();

    await userEvent.type(
      screen.getByLabelText(/trace url/i),
      'http://x/y.json',
    );
    expect(loadButton).toBeEnabled();
  });
});
