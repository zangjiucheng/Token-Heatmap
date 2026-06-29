/**
 * Render a token string with whitespace and ASCII control characters
 * visible: newline → "\\n", tab → "\\t", other 0x00-0x1F / 0x7F → "\\xNN".
 * Printable ASCII and non-ASCII are passed through unchanged.
 */
export function escapeToken(token: string): string {
  let out = '';
  for (let i = 0; i < token.length; i += 1) {
    const ch = token[i];
    const code = ch.charCodeAt(0);
    if (ch === '\n') {
      out += '\\n';
    } else if (ch === '\t') {
      out += '\\t';
    } else if (ch === '\r') {
      out += '\\r';
    } else if (ch === '\\') {
      out += '\\\\';
    } else if (code < 0x20 || code === 0x7f) {
      out += `\\x${code.toString(16).padStart(2, '0').toUpperCase()}`;
    } else {
      out += ch;
    }
  }
  return out;
}
