/**
 * Centralized logging utility with consistent formatting
 */

export interface Logger {
  info(message: string, ...args: any[]): void
  error(message: string, ...args: any[]): void
  warn(message: string, ...args: any[]): void
  debug(message: string, ...args: any[]): void
}

export function createLogger(prefix: string): Logger {
  return {
    info: (message: string, ...args: any[]) => {
      console.log(`${prefix} ${message}`, ...args)
    },
    
    error: (message: string, ...args: any[]) => {
      console.error(`${prefix} ❌ ${message}`, ...args)
    },
    
    warn: (message: string, ...args: any[]) => {
      console.warn(`${prefix} ⚠️ ${message}`, ...args)
    },
    
    debug: (message: string, ...args: any[]) => {
      console.log(`${prefix} 🐛 ${message}`, ...args)
    }
  }
}

// Pre-configured loggers for common areas
export const sessionLogger = createLogger('🎙️ [SESSION]')
export const audioLogger = createLogger('🎤 [AUDIO]')
export const utteranceLogger = createLogger('🔚 [UTTERANCE]')
export const speakerLogger = createLogger('🔍 [SPEAKER_ID]')
export const transcriptLogger = createLogger('📝 [UTTERANCE]')
export const deepgramLogger = createLogger('🌐 [DEEPGRAM]')