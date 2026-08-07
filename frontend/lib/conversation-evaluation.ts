import type { ChatEvaluation, ChatHistoryRecord } from '@/lib/chat-history'
import { pb } from '@/lib/pocketbase'

const apiEndpoint = import.meta.env.VITE_API_ENDPOINT ?? 'http://127.0.0.1:8000'

export const canEvaluateChatHistory = (
  record: ChatHistoryRecord,
  activeHistoryId: string | null,
  chatStatus: 'submitted' | 'streaming' | 'ready' | 'error'
) =>
  record.status === 'active' &&
  record.conversation.length > 0 &&
  !(
    record.id === activeHistoryId &&
    (chatStatus === 'submitted' || chatStatus === 'streaming')
  )

export const getEvaluationConfig = async () => {
  const response = await fetch(`${apiEndpoint.replace(/\/$/, '')}/evaluation-config`)
  if (!response.ok) {
    return { cloudEvaluation: false }
  }
  const data: unknown = await response.json()
  return {
    cloudEvaluation:
      typeof data === 'object' &&
      data !== null &&
      'cloud_evaluation' in data &&
      data.cloud_evaluation === true,
  }
}

type EvaluationEvent = {
  event: string
  data: unknown
}

const parseEvent = (chunk: string): EvaluationEvent | null => {
  const event = chunk.match(/^event: (.+)$/m)?.[1]
  const dataText = chunk.match(/^data: (.+)$/m)?.[1]
  if (!event || !dataText) {
    return null
  }
  try {
    return { event, data: JSON.parse(dataText) }
  } catch {
    return null
  }
}

export const evaluateChatHistory = async (
  chatHistoryId: string,
  ollamaModel: string,
  handlers: {
    onError: (message: string) => void
    onProgress: (progress: number, message: string) => void
    onResult: (evaluation: ChatEvaluation, completedAt: string) => void
  }
) => {
  const response = await fetch(
    `${apiEndpoint.replace(/\/$/, '')}/chat-history/${chatHistoryId}/evaluate`,
    {
      body: JSON.stringify({ ollama_model: ollamaModel }),
      headers: {
        Authorization: `Bearer ${pb.authStore.token}`,
        'Content-Type': 'application/json',
      },
      method: 'POST',
    }
  )

  if (!response.ok || !response.body) {
    throw new Error('Could not start the evaluation.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let receivedResult = false

  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const text of events) {
      const event = parseEvent(text)
      if (!event || typeof event.data !== 'object' || event.data === null) {
        continue
      }
      const data = event.data as Record<string, unknown>
      if (event.event === 'progress') {
        handlers.onProgress(
          typeof data.progress === 'number' ? data.progress : 0,
          typeof data.message === 'string' ? data.message : 'Evaluating conversation'
        )
      } else if (event.event === 'result' && typeof data.completed_at === 'string') {
        receivedResult = true
        handlers.onResult(data.evaluation as ChatEvaluation, data.completed_at)
      } else if (event.event === 'error') {
        handlers.onError(
          typeof data.message === 'string' ? data.message : 'Evaluation failed. Please try again.'
        )
      }
    }

    if (done) {
      break
    }
  }

  if (!receivedResult) {
    throw new Error('The evaluation ended before a result was returned.')
  }
}
