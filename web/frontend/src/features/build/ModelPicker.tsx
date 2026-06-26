import {
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react';
import {
  MODEL_PRESET_GROUPS,
  MODEL_PRESETS,
  modelShortLabel,
  modelSizeLabel,
} from './config';
import './ModelPicker.css';

export interface ModelPickerProps {
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
}

/**
 * A searchable model combobox: type to filter the presets (or paste any HF
 * causal-LM id), pick from a grouped, size-badged list. Replaces the old
 * free-text box + wall of preset buttons. Size badges hint how heavy a model
 * is to load and run, so it's easy to pick a fast one.
 */
export function ModelPicker({ value, onChange, disabled }: ModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const listId = useId();

  // When the field holds a known preset (or is empty) we're browsing, so show
  // everything; once the user types something else we filter by substring.
  const isPreset = MODEL_PRESETS.includes(value);
  const query = value.trim().toLowerCase();

  const groups = useMemo(
    () =>
      MODEL_PRESET_GROUPS.map((g) => ({
        family: g.family,
        models: g.models.filter(
          (m) =>
            isPreset ||
            query === '' ||
            m.toLowerCase().includes(query) ||
            modelShortLabel(m).toLowerCase().includes(query),
        ),
      })).filter((g) => g.models.length > 0),
    [isPreset, query],
  );

  const flat = useMemo(() => groups.flatMap((g) => g.models), [groups]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  useEffect(() => {
    setActiveIndex((i) => (i >= flat.length ? flat.length - 1 : i));
  }, [flat.length]);

  // The dropdown is `position: fixed` so it escapes the Build rail's
  // overflow-x:auto (which clips overflowing descendants, cutting the list off).
  // Anchor it to the field's rect, tracking scroll/resize while open.
  const [popStyle, setPopStyle] = useState<CSSProperties | null>(null);
  useLayoutEffect(() => {
    if (!open) {
      setPopStyle(null);
      return undefined;
    }
    const update = () => {
      const r = rootRef.current?.getBoundingClientRect();
      if (r) setPopStyle({ top: r.bottom + 4, left: r.left, width: r.width });
    };
    update();
    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
    };
  }, [open]);

  const select = (model: string) => {
    onChange(model);
    setOpen(false);
    setActiveIndex(-1);
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) {
        setOpen(true);
        return;
      }
      setActiveIndex((i) => Math.min(flat.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter') {
      if (open && activeIndex >= 0 && flat[activeIndex]) {
        e.preventDefault();
        select(flat[activeIndex]);
      }
    } else if (e.key === 'Escape') {
      if (open) {
        e.preventDefault();
        setOpen(false);
      }
    }
  };

  const currentSize = modelSizeLabel(value);
  const optionId = (i: number) => `${listId}-opt-${i}`;

  return (
    <div className="model-picker" ref={rootRef}>
      <div className="model-picker__field">
        <input
          type="text"
          role="combobox"
          aria-label="Model"
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={
            open && activeIndex >= 0 ? optionId(activeIndex) : undefined
          }
          className="node-field__input model-picker__input"
          value={value}
          disabled={disabled}
          placeholder="Search models or paste any HF id…"
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
            setActiveIndex(-1);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          data-testid="node-input-model"
        />
        {currentSize ? (
          <span className="model-picker__size" aria-hidden="true">
            {currentSize}
          </span>
        ) : null}
        <button
          type="button"
          className="model-picker__toggle"
          tabIndex={-1}
          aria-label={open ? 'Hide model list' : 'Show model list'}
          disabled={disabled}
          onClick={() => setOpen((o) => !o)}
        >
          ▾
        </button>
      </div>

      {open ? (
        <div
          className="model-picker__pop"
          style={popStyle ?? { visibility: 'hidden' }}
        >
          <p className="model-picker__note">
            Smaller models load &amp; run faster — or paste any Hugging Face
            causal-LM id.
          </p>
          <div
            className="model-picker__list"
            role="listbox"
            id={listId}
            aria-label="Model presets"
          >
            {flat.length === 0 ? (
              <div className="model-picker__empty">
                No preset matches — “{value.trim()}” will be used as a custom
                model id.
              </div>
            ) : (
              groups.map((g) => (
                <div
                  key={g.family}
                  role="group"
                  aria-label={g.family}
                  className="model-picker__group"
                >
                  <div className="model-picker__group-label eyebrow">
                    {g.family}
                  </div>
                  {g.models.map((m) => {
                    const idx = flat.indexOf(m);
                    const selected = m === value;
                    const size = modelSizeLabel(m);
                    return (
                      <button
                        key={m}
                        type="button"
                        role="option"
                        id={optionId(idx)}
                        aria-selected={selected}
                        className={[
                          'model-picker__option',
                          idx === activeIndex
                            ? 'model-picker__option--active'
                            : '',
                          selected ? 'model-picker__option--selected' : '',
                        ]
                          .filter(Boolean)
                          .join(' ')}
                        title={m}
                        onMouseDown={(e) => e.preventDefault()}
                        onMouseEnter={() => setActiveIndex(idx)}
                        onClick={() => select(m)}
                      >
                        <span className="model-picker__option-label">
                          {modelShortLabel(m)}
                        </span>
                        {size ? (
                          <span className="model-picker__option-size">
                            {size}
                          </span>
                        ) : null}
                      </button>
                    );
                  })}
                </div>
              ))
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
