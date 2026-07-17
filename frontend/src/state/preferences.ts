import { backgroundOptions, defaultBackgroundId } from '@/lib/backgrounds'
import { create } from 'zustand'

const backgroundStorageKey = 'p-gpt-background'
const parametersStorageKey = 'p-gpt-generation-parameters'

export type GenerationParameters = {
  temperature: number
  cloneVoice: boolean
  maxNewTokens: number
  repeatPenalty: 1 | 1.1 | 1.2
  seed: number | null
}

export const defaultGenerationParameters: GenerationParameters = {
  temperature: 1,
  cloneVoice: true,
  maxNewTokens: 256,
  repeatPenalty: 1,
  seed: null,
}

const getInitialBackgroundId = () => {
  try {
    const storedBackgroundId = window.localStorage.getItem(backgroundStorageKey)
    if (
      storedBackgroundId &&
      backgroundOptions.some((option) => option.id === storedBackgroundId)
    ) {
      return storedBackgroundId
    }
  } catch {
    // Local storage can be unavailable in privacy-restricted browsers.
  }

  return defaultBackgroundId
}

const getInitialGenerationParameters = (): GenerationParameters => {
  try {
    const storedValue = window.localStorage.getItem(parametersStorageKey)
    if (!storedValue) {
      return defaultGenerationParameters
    }

    const parsed: unknown = JSON.parse(storedValue)
    if (typeof parsed !== 'object' || parsed === null) {
      return defaultGenerationParameters
    }

    const temperature =
      'temperature' in parsed && typeof parsed.temperature === 'number'
        ? Math.min(2, Math.max(0, parsed.temperature))
        : defaultGenerationParameters.temperature
    const cloneVoice =
      'cloneVoice' in parsed && typeof parsed.cloneVoice === 'boolean'
        ? parsed.cloneVoice
        : defaultGenerationParameters.cloneVoice
    const maxNewTokens =
      'maxNewTokens' in parsed && typeof parsed.maxNewTokens === 'number'
        ? Math.min(8192, Math.max(64, Math.round(parsed.maxNewTokens)))
        : defaultGenerationParameters.maxNewTokens
    const repeatPenalty =
      'repeatPenalty' in parsed &&
      (parsed.repeatPenalty === 1 ||
        parsed.repeatPenalty === 1.1 ||
        parsed.repeatPenalty === 1.2)
        ? parsed.repeatPenalty
        : defaultGenerationParameters.repeatPenalty
    const seed =
      'seed' in parsed &&
      typeof parsed.seed === 'number' &&
      Number.isSafeInteger(parsed.seed)
        ? parsed.seed
        : defaultGenerationParameters.seed

    return {
      cloneVoice,
      maxNewTokens,
      repeatPenalty,
      seed,
      temperature,
    }
  } catch {
    return defaultGenerationParameters
  }
}

interface PreferencesState {
  selectedBackgroundId: string
  generationParameters: GenerationParameters
  selectBackground: (backgroundId: string) => void
  setGenerationParameter: <Key extends keyof GenerationParameters>(
    key: Key,
    value: GenerationParameters[Key]
  ) => void
  resetGenerationParameters: () => void
}

const persistGenerationParameters = (parameters: GenerationParameters) => {
  try {
    window.localStorage.setItem(parametersStorageKey, JSON.stringify(parameters))
  } catch {
    // The in-memory preferences still apply when storage is unavailable.
  }
}

export const usePreferencesStore = create<PreferencesState>((set) => ({
  selectedBackgroundId: getInitialBackgroundId(),
  generationParameters: getInitialGenerationParameters(),
  selectBackground: (backgroundId) => {
    if (!backgroundOptions.some((option) => option.id === backgroundId)) {
      return
    }

    try {
      window.localStorage.setItem(backgroundStorageKey, backgroundId)
    } catch {
      // The in-memory preference still applies when storage is unavailable.
    }

    set({ selectedBackgroundId: backgroundId })
  },
  setGenerationParameter: (key, value) => {
    set((state) => {
      const generationParameters = {
        ...state.generationParameters,
        [key]: value,
      }
      persistGenerationParameters(generationParameters)
      return { generationParameters }
    })
  },
  resetGenerationParameters: () => {
    persistGenerationParameters(defaultGenerationParameters)
    set({ generationParameters: defaultGenerationParameters })
  },
}))
