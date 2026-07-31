import { getApiErrorMessage } from '@/lib/api-error'

const apiEndpoint =
  import.meta.env.VITE_API_ENDPOINT ?? 'http://127.0.0.1:8000'

export type OllamaModels = {
  defaultModel: string
  models: string[]
  usedFallback: boolean
}

export const listOllamaModels = async (): Promise<OllamaModels> => {
  const response = await fetch(`${apiEndpoint}/ollama/models`, {
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    throw new Error(await getApiErrorMessage(response, 'Model discovery failed'))
  }

  const payload: unknown = await response.json()
  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('models' in payload) ||
    !Array.isArray(payload.models) ||
    !payload.models.every((model) => typeof model === 'string') ||
    !('default_model' in payload) ||
    typeof payload.default_model !== 'string' ||
    !('used_fallback' in payload) ||
    typeof payload.used_fallback !== 'boolean'
  ) {
    throw new Error('Model discovery returned an invalid response.')
  }

  return {
    defaultModel: payload.default_model,
    models: payload.models,
    usedFallback: payload.used_fallback,
  }
}
