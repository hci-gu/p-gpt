import { backgroundOptions, defaultBackgroundId } from '@/lib/backgrounds'
import { asrModels, defaultAsrModel, type AsrModel } from '@/src/lib/asr-models'
import { create } from 'zustand'

const backgroundStorageKey = 'p-gpt-background'
const parametersStorageKey = 'p-gpt-generation-parameters'
export const conversationModels = ['gemma4:e4b', 'gemma4:e2b', 'gemma4:31b'] as const
export type ConversationModel = (typeof conversationModels)[number]
export const defaultConversationModel: ConversationModel = 'gemma4:e4b'

export type GenerationParameters = {
  asrModel: AsrModel
  model: ConversationModel
  temperature: number
  cloneVoice: boolean
  maxNewTokens: number
  streaming: boolean
  ttsStepLevel: number
  repeatPenalty: 1 | 1.1 | 1.2
  seed: number | null
}

export const omnivoiceNumStepsFromLevel = (level: number) =>
  Math.round(22 + ((Math.min(10, Math.max(1, level)) - 1) * 10) / 9)

export const defaultGenerationParameters: GenerationParameters = {
  asrModel: defaultAsrModel,
  model: defaultConversationModel,
  temperature: 1,
  cloneVoice: true,
  maxNewTokens: 256,
  streaming: false,
  ttsStepLevel: 5,
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
    const model =
      'model' in parsed &&
      typeof parsed.model === 'string' &&
      conversationModels.includes(parsed.model as ConversationModel)
        ? (parsed.model as ConversationModel)
        : defaultGenerationParameters.model
    const asrModel =
      'asrModel' in parsed &&
      typeof parsed.asrModel === 'string' &&
      asrModels.includes(parsed.asrModel as AsrModel)
        ? (parsed.asrModel as AsrModel)
        : defaultGenerationParameters.asrModel
    const cloneVoice =
      'cloneVoice' in parsed && typeof parsed.cloneVoice === 'boolean'
        ? parsed.cloneVoice
        : defaultGenerationParameters.cloneVoice
    const maxNewTokens =
      'maxNewTokens' in parsed && typeof parsed.maxNewTokens === 'number'
        ? Math.min(8192, Math.max(64, Math.round(parsed.maxNewTokens)))
        : defaultGenerationParameters.maxNewTokens
    const streaming =
      'streaming' in parsed && typeof parsed.streaming === 'boolean'
        ? parsed.streaming
        : defaultGenerationParameters.streaming
    const ttsStepLevel =
      'ttsStepLevel' in parsed && typeof parsed.ttsStepLevel === 'number'
        ? Math.min(10, Math.max(1, Math.round(parsed.ttsStepLevel)))
        : defaultGenerationParameters.ttsStepLevel
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
      Number.isSafeInteger(parsed.seed) &&
      parsed.seed >= 0
        ? parsed.seed
        : defaultGenerationParameters.seed

    return {
      asrModel,
      cloneVoice,
      maxNewTokens,
      model,
      repeatPenalty,
      seed,
      streaming,
      temperature,
      ttsStepLevel,
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
