declare const process: { env?: Record<string, string | undefined> } | undefined;

export function alignTo(x: number, align: number): number {
  return Math.floor((x + align - 1) / align) * align;
}

export function _debugAssert(x: boolean, msg: string) {
  if (!x) {
    throw msg
  }
}

export function _assert(x: boolean, msg: string) {
  if (!x) {
    throw msg
  }
}

type DebugLevel = 'log' | 'info' | 'warn';

let cachedDebugFlag: boolean | null = null;

function readDebugPreference(): boolean {
  if (typeof globalThis !== 'undefined') {
    const anyGlobal = globalThis as Record<string, unknown>;
    const direct = anyGlobal.WEBRTX_DEBUG_LOGGING;
    if (typeof direct === 'boolean') {
      return direct;
    }
  }
  if (typeof process !== 'undefined' && typeof process.env?.WEBRTX_DEBUG_LOGGING === 'string') {
    const value = process.env.WEBRTX_DEBUG_LOGGING.trim();
    if (value.length > 0) {
      return value !== '0' && value.toLowerCase() !== 'false';
    }
  }
  if (typeof localStorage !== 'undefined') {
    const stored = localStorage.getItem('WEBRTX_DEBUG_LOGGING');
    if (stored !== null) {
      return stored !== '0' && stored.toLowerCase() !== 'false';
    }
  }
  return false;
}

export function isWebrtxDebugLoggingEnabled(): boolean {
  if (cachedDebugFlag === null) {
    cachedDebugFlag = readDebugPreference();
  }
  return cachedDebugFlag;
}

export function setWebrtxDebugLoggingEnabled(enabled: boolean): void {
  cachedDebugFlag = enabled;
  if (typeof globalThis !== 'undefined') {
    (globalThis as Record<string, unknown>).WEBRTX_DEBUG_LOGGING = enabled;
  }
  if (typeof localStorage !== 'undefined') {
    try {
      localStorage.setItem('WEBRTX_DEBUG_LOGGING', enabled ? '1' : '0');
    } catch {
      /* ignore storage failures */
    }
  }
}

function emitDebugMessage(level: DebugLevel, args: unknown[]): void {
  if (!isWebrtxDebugLoggingEnabled()) {
    return;
  }
  if (typeof console === 'undefined') {
    return;
  }
  const logger = console[level] ?? console.log;
  logger.apply(console, args as []);
}

export function debugLog(...args: unknown[]): void {
  emitDebugMessage('log', args);
}

export function debugInfo(...args: unknown[]): void {
  emitDebugMessage('info', args);
}

export function debugWarn(...args: unknown[]): void {
  emitDebugMessage('warn', args);
}