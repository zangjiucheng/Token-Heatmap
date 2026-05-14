import { describe, expect, it } from 'vitest';
import { escapeToken } from './escapeToken';

describe('escapeToken', () => {
  it('converts newline to "\\n"', () => {
    expect(escapeToken('\n')).toBe('\\n');
  });

  it('converts tab to "\\t"', () => {
    expect(escapeToken('\t')).toBe('\\t');
  });

  it('converts carriage return to "\\r"', () => {
    expect(escapeToken('\r')).toBe('\\r');
  });

  it('converts NUL to "\\x00"', () => {
    expect(escapeToken('\x00')).toBe('\\x00');
  });

  it('converts DEL (0x7F) to "\\x7F"', () => {
    expect(escapeToken('\x7f')).toBe('\\x7F');
  });

  it('converts other ASCII control chars to "\\xNN" with uppercase hex', () => {
    expect(escapeToken('\x01')).toBe('\\x01');
    expect(escapeToken('\x1b')).toBe('\\x1B');
  });

  it('escapes backslashes so output round-trips visually', () => {
    expect(escapeToken('\\n')).toBe('\\\\n');
  });

  it('passes regular printable ASCII through unchanged', () => {
    expect(escapeToken('hello world')).toBe('hello world');
    expect(escapeToken('a-z_0-9!?')).toBe('a-z_0-9!?');
  });

  it('passes non-ASCII (unicode) through unchanged', () => {
    expect(escapeToken('Paris')).toBe('Paris');
    expect(escapeToken('café')).toBe('café');
    expect(escapeToken('日本語')).toBe('日本語');
    expect(escapeToken('🦀')).toBe('🦀');
  });

  it('handles mixed content', () => {
    expect(escapeToken('line1\nline2\tend')).toBe('line1\\nline2\\tend');
  });

  it('handles the empty string', () => {
    expect(escapeToken('')).toBe('');
  });
});
