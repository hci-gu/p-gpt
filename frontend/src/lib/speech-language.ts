export const speechLanguages = ['en', 'sv'] as const

export type SpeechLanguage = (typeof speechLanguages)[number]

export const isSpeechLanguage = (value: unknown): value is SpeechLanguage =>
  typeof value === 'string' &&
  speechLanguages.includes(value as SpeechLanguage)
