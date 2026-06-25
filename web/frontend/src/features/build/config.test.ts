import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BUILD_CONFIG,
  buildConfigToParams,
  buildConfigToYaml,
  isBuildConfigValid,
  type BuildConfig,
} from './config';

const config: BuildConfig = {
  ...DEFAULT_BUILD_CONFIG,
  model: '  Qwen/Qwen2.5-7B-Instruct  ',
  prompt: 'line one\nline two',
  max_new_tokens: 128,
  capture_attention: true,
  out: 'outputs/demo',
};

describe('buildConfigToParams', () => {
  it('trims the model and maps every generate field', () => {
    const params = buildConfigToParams(config);
    expect(params.model).toBe('Qwen/Qwen2.5-7B-Instruct');
    expect(params.prompt).toBe('line one\nline two');
    expect(params.max_new_tokens).toBe(128);
    expect(params.capture_attention).toBe(true);
    // `out` is CLI-only and must not leak into the request body.
    expect('out' in params).toBe(false);
  });
});

describe('isBuildConfigValid', () => {
  it('requires both a model and a prompt', () => {
    expect(isBuildConfigValid(config)).toBe(true);
    expect(isBuildConfigValid({ ...config, prompt: '   ' })).toBe(false);
    expect(isBuildConfigValid({ ...config, model: '' })).toBe(false);
  });
});

describe('buildConfigToYaml', () => {
  it('emits a CLI-loadable config with safely-quoted multi-line prompt', () => {
    const yaml = buildConfigToYaml(config);
    expect(yaml).toContain('model: "Qwen/Qwen2.5-7B-Instruct"');
    // Newlines in the prompt are escaped inside a double-quoted scalar.
    expect(yaml).toContain('prompt: "line one\\nline two"');
    expect(yaml).toContain('max_new_tokens: 128');
    expect(yaml).toContain('out: "outputs/demo"');
    expect(yaml).toContain('capture_attention: true');
    expect(yaml).toContain('capture_logit_lens: false');
  });

  it('falls back to a default output dir when blank', () => {
    const yaml = buildConfigToYaml({ ...config, out: '   ' });
    expect(yaml).toContain('out: "outputs/run"');
  });
});
