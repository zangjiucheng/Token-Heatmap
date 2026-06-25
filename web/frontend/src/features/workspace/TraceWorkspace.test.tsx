import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { TraceWorkspace } from './TraceWorkspace';

function renderWorkspace(
  overrides: Partial<Parameters<typeof TraceWorkspace>[0]> = {},
) {
  const onToggleInspector = vi.fn();
  render(
    <TraceWorkspace
      model="Qwen/Qwen2.5-7B"
      prompt="The quick brown fox"
      stepCount={12}
      selectedStep={3}
      tokenStrip={<div data-testid="spine-strip">strip</div>}
      timelines={<div data-testid="spine-timelines">timelines</div>}
      rail={<div data-testid="the-rail">rail</div>}
      controlBar={<div data-testid="the-controls">controls</div>}
      canvas={<div data-testid="the-canvas">canvas</div>}
      inspector={<div data-testid="the-inspector">inspector</div>}
      inspectorOpen
      onToggleInspector={onToggleInspector}
      railCollapsed={false}
      {...overrides}
    />,
  );
  return { onToggleInspector };
}

describe('TraceWorkspace', () => {
  it('shows trace identity in the sub-header', () => {
    renderWorkspace();
    expect(
      screen.getByRole('heading', { name: 'Qwen/Qwen2.5-7B' }),
    ).toBeInTheDocument();
    expect(screen.getByText('The quick brown fox')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('Step 3')).toBeInTheDocument();
  });

  it('renders spine, rail, controls, canvas and inspector slots', () => {
    renderWorkspace();
    expect(screen.getByTestId('spine-strip')).toBeInTheDocument();
    expect(screen.getByTestId('spine-timelines')).toBeInTheDocument();
    expect(screen.getByTestId('the-rail')).toBeInTheDocument();
    expect(screen.getByTestId('the-controls')).toBeInTheDocument();
    expect(screen.getByTestId('the-canvas')).toBeInTheDocument();
    expect(screen.getByTestId('the-inspector')).toBeInTheDocument();
  });

  it('omits the controls slot when no control bar is given', () => {
    renderWorkspace({ controlBar: undefined });
    expect(screen.queryByTestId('the-controls')).not.toBeInTheDocument();
  });

  it('collapses the inspector to an expand affordance', async () => {
    const { onToggleInspector } = renderWorkspace({ inspectorOpen: false });
    expect(screen.queryByTestId('the-inspector')).not.toBeInTheDocument();
    await userEvent.click(
      screen.getByRole('button', { name: /show inspector/i }),
    );
    expect(onToggleInspector).toHaveBeenCalledTimes(1);
  });

  it('hides the inspector via the collapse button when open', async () => {
    const { onToggleInspector } = renderWorkspace();
    await userEvent.click(
      screen.getByRole('button', { name: /hide inspector/i }),
    );
    expect(onToggleInspector).toHaveBeenCalledTimes(1);
  });

  it('collapses the overview timelines, keeping the toggle reachable', async () => {
    const onToggleTimelines = vi.fn();
    renderWorkspace({ timelinesOpen: false, onToggleTimelines });
    // The timelines themselves are hidden, but the toggle bar remains.
    expect(screen.queryByTestId('spine-timelines')).not.toBeInTheDocument();
    await userEvent.click(screen.getByTestId('overview-toggle'));
    expect(onToggleTimelines).toHaveBeenCalledTimes(1);
  });
});
