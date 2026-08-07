export type SpeakerVadDetectionProfile = 'barge-in' | 'start'

export type SpeakerVadActivationConfig = {
  requiredFrames: number
  threshold: number
  windowFrames: number
}

export type SpeakerVadConfig = {
  bargeIn: SpeakerVadActivationConfig
  continueThreshold: number
  preRollFrames: number
  softEndFrames: number
  start: SpeakerVadActivationConfig
}

export const defaultSpeakerVadConfig: SpeakerVadConfig = {
  bargeIn: {
    requiredFrames: 4,
    threshold: 0.4,
    windowFrames: 6,
  },
  continueThreshold: 0.3,
  preRollFrames: 24,
  softEndFrames: 12,
  start: {
    requiredFrames: 4,
    threshold: 0.4,
    windowFrames: 6,
  },
}

type SpeakerVadEnvironment = Record<string, string | boolean | undefined>

const parseFiniteNumber = (
  environment: SpeakerVadEnvironment,
  key: string,
  fallback: number,
  minimum: number,
  maximum: number,
  integer: boolean,
  warn: (message: string) => void
) => {
  const rawValue = environment[key]
  if (rawValue === undefined || rawValue === '') {
    return fallback
  }

  const value = typeof rawValue === 'string' ? Number(rawValue) : Number.NaN
  if (
    !Number.isFinite(value) ||
    value < minimum ||
    value > maximum ||
    (integer && !Number.isInteger(value))
  ) {
    warn(`[speaker-vad] Invalid ${key}=${String(rawValue)}; using ${fallback}.`)
    return fallback
  }
  return value
}

const parseActivationConfig = (
  environment: SpeakerVadEnvironment,
  prefix: string,
  fallback: SpeakerVadActivationConfig,
  warn: (message: string) => void
): SpeakerVadActivationConfig => {
  const threshold = parseFiniteNumber(
    environment,
    `${prefix}_THRESHOLD`,
    fallback.threshold,
    0.05,
    0.95,
    false,
    warn
  )
  const requiredFrames = parseFiniteNumber(
    environment,
    `${prefix}_REQUIRED_FRAMES`,
    fallback.requiredFrames,
    1,
    30,
    true,
    warn
  )
  const windowFrames = parseFiniteNumber(
    environment,
    `${prefix}_WINDOW_FRAMES`,
    fallback.windowFrames,
    1,
    30,
    true,
    warn
  )

  if (requiredFrames > windowFrames) {
    warn(
      `[speaker-vad] ${prefix}_REQUIRED_FRAMES cannot exceed ${prefix}_WINDOW_FRAMES; using defaults.`
    )
    return { ...fallback }
  }
  return { requiredFrames, threshold, windowFrames }
}

export const parseSpeakerVadConfig = (
  environment: SpeakerVadEnvironment,
  warn: (message: string) => void = console.warn
): SpeakerVadConfig => ({
  bargeIn: parseActivationConfig(
    environment,
    'VITE_SPEAKER_VAD_BARGE',
    defaultSpeakerVadConfig.bargeIn,
    warn
  ),
  continueThreshold: parseFiniteNumber(
    environment,
    'VITE_SPEAKER_VAD_CONTINUE_THRESHOLD',
    defaultSpeakerVadConfig.continueThreshold,
    0.05,
    0.95,
    false,
    warn
  ),
  preRollFrames: parseFiniteNumber(
    environment,
    'VITE_SPEAKER_VAD_PRE_ROLL_FRAMES',
    defaultSpeakerVadConfig.preRollFrames,
    1,
    120,
    true,
    warn
  ),
  softEndFrames: parseFiniteNumber(
    environment,
    'VITE_SPEAKER_VAD_SOFT_END_FRAMES',
    defaultSpeakerVadConfig.softEndFrames,
    1,
    120,
    true,
    warn
  ),
  start: parseActivationConfig(
    environment,
    'VITE_SPEAKER_VAD_START',
    defaultSpeakerVadConfig.start,
    warn
  ),
})

export const speakerVadConfig = parseSpeakerVadConfig(import.meta.env)

export const getSpeakerVadActivationConfig = (
  config: SpeakerVadConfig,
  profile: SpeakerVadDetectionProfile
) => (profile === 'barge-in' ? config.bargeIn : config.start)

export const describeSpeakerVadConfig = (config: SpeakerVadConfig) =>
  [
    `start=${config.start.threshold}:${config.start.requiredFrames}/${config.start.windowFrames}`,
    `barge=${config.bargeIn.threshold}:${config.bargeIn.requiredFrames}/${config.bargeIn.windowFrames}`,
    `continue=${config.continueThreshold}`,
    `pre_roll_frames=${config.preRollFrames}`,
    `soft_end_frames=${config.softEndFrames}`,
  ].join(';')
