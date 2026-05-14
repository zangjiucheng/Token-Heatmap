import '@testing-library/jest-dom/vitest';

// Some Node/jsdom combinations expose a partial localStorage implementation
// (getItem/setItem exist, but clear/removeItem do not). Normalize it once so
// tests exercise browser-like storage behavior consistently.
if (typeof window !== 'undefined') {
  const storage = new Map<string, string>();
  const localStorageMock: Storage = {
    get length() {
      return storage.size;
    },
    clear() {
      storage.clear();
    },
    getItem(key: string) {
      return storage.has(key) ? storage.get(key)! : null;
    },
    key(index: number) {
      return Array.from(storage.keys())[index] ?? null;
    },
    removeItem(key: string) {
      storage.delete(key);
    },
    setItem(key: string, value: string) {
      storage.set(key, String(value));
    },
  };
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: localStorageMock,
  });
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: localStorageMock,
  });
}

// jsdom 24 does not implement the PointerEvent constructor, so React's
// synthetic onPointerDown / onPointerMove / onPointerUp handlers never fire
// from fireEvent.pointerDown(...). Polyfill it as a thin MouseEvent subclass
// that preserves clientX / clientY / pointerId / pointerType from the init
// dictionary — that's all our components read from the event.
if (
  typeof window !== 'undefined' &&
  typeof (window as unknown as { PointerEvent?: unknown }).PointerEvent ===
    'undefined'
) {
  class PolyfilledPointerEvent extends MouseEvent {
    pointerId: number;
    pointerType: string;
    isPrimary: boolean;
    width: number;
    height: number;
    pressure: number;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 0;
      this.pointerType = init.pointerType ?? '';
      this.isPrimary = init.isPrimary ?? false;
      this.width = init.width ?? 1;
      this.height = init.height ?? 1;
      this.pressure = init.pressure ?? 0;
    }
  }
  (
    window as unknown as { PointerEvent: typeof PolyfilledPointerEvent }
  ).PointerEvent = PolyfilledPointerEvent;
  (
    globalThis as unknown as { PointerEvent: typeof PolyfilledPointerEvent }
  ).PointerEvent = PolyfilledPointerEvent;
}

// jsdom Elements lack setPointerCapture / releasePointerCapture. Stub them
// so component handlers calling these methods do not throw.
if (
  typeof Element !== 'undefined' &&
  typeof (Element.prototype as unknown as { setPointerCapture?: unknown })
    .setPointerCapture !== 'function'
) {
  (
    Element.prototype as unknown as { setPointerCapture: () => void }
  ).setPointerCapture = function setPointerCapture() {
    /* no-op for jsdom */
  };
  (
    Element.prototype as unknown as { releasePointerCapture: () => void }
  ).releasePointerCapture = function releasePointerCapture() {
    /* no-op for jsdom */
  };
  (
    Element.prototype as unknown as { hasPointerCapture: () => boolean }
  ).hasPointerCapture = function hasPointerCapture() {
    return false;
  };
}

// jsdom's Blob/File implementations omit `text()` and `arrayBuffer()`.
// Patch them in so loaders that read user-dropped files behave like a
// real browser under test.
if (typeof Blob.prototype.text !== 'function') {
  Blob.prototype.text = function text(this: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(this);
    });
  };
}

// jsdom does not implement HTMLCanvasElement.getContext. The heatmap renders
// to canvas in real browsers; under test we just need it to not throw so
// React effects can complete. Components handle a null context.
if (
  typeof HTMLCanvasElement !== 'undefined' &&
  !('__patchedGetContext' in HTMLCanvasElement.prototype)
) {
  Object.defineProperty(HTMLCanvasElement.prototype, '__patchedGetContext', {
    value: true,
  });
  HTMLCanvasElement.prototype.getContext = function getContext() {
    return null;
  } as typeof HTMLCanvasElement.prototype.getContext;
}
