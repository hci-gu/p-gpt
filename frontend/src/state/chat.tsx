'use client'

import type { ToolUIPart } from 'ai'
import { create } from 'zustand'

const apiEndpoint = import.meta.env.VITE_API_ENDPOINT ?? ''
const chatInitializeEndpoint = `${apiEndpoint}/initiate-request`
const chatTextEndpoint = (chatId: string) =>
  `${apiEndpoint}/requests/${chatId}/text`
const chatTtsStreamEndpoint = (chatId: string) =>
  `${apiEndpoint}/requests/${chatId}/audio`
const chatInterruptEndpoint = (chatId: string) =>
  `${apiEndpoint}/requests/${chatId}/interrupt`

export interface MessageType {
  key: string
  from: 'user' | 'assistant' | 'system'
  sources?: { href: string; title: string }[]
  versions: {
    id: string
    content: string
    contentStatus?: 'pending' | 'ready' | 'error'
    audioPlaybackComplete?: boolean
    audioUrl?: string
  }[]
  reasoning?: {
    content: string
    duration: number
  }
  tools?: {
    name: string
    description: string
    status: ToolUIPart['state']
    parameters: Record<string, unknown>
    result: string | undefined
    error: string | undefined
  }[]
}

export const initialMessages: MessageType[] = []

export const suggestions = [
  'Who are you?',
  'Tell me your top 5 favorite movies',
  'What did you do yesterday?',
  "Give me 5 bulletpoints list on what to bring on a camping trip and why."
]

export type ChatStatus = 'submitted' | 'streaming' | 'ready' | 'error'

type ChatHistoryMessage = {
  role: MessageType['from']
  content: string
}

const createMessageId = (prefix: string) =>
  `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`

const toChatHistory = (messages: MessageType[]): ChatHistoryMessage[] =>
  messages.flatMap((message) =>
    message.versions
      .map((version) => ({
        content: version.content.trim(),
        role: message.from,
      }))
      .filter((message) => message.content)
  )

const initializeChat = async (
  messages: ChatHistoryMessage[],
  signal?: AbortSignal
) => {
  const response = await fetch(chatInitializeEndpoint, {
    body: JSON.stringify({
      messages,
      response_format: 'pcm',
      stream_audio: true,
    }),
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    method: 'POST',
    signal,
  })

  if (!response.ok) {
    throw new Error(`Chat initialization failed with status ${response.status}`)
  }

  const data: unknown = await response.json()

  if (
    typeof data === 'object' &&
    data !== null &&
    'request_id' in data &&
    typeof data.request_id === 'string'
  ) {
    return data.request_id
  }

  if (
    typeof data === 'object' &&
    data !== null &&
    'id' in data &&
    typeof data.id === 'string'
  ) {
    return data.id
  }

  throw new Error('Chat initialization response did not include an id.')
}

const fetchTextResponse = async (chatId: string, signal?: AbortSignal) => {
  const response = await fetch(chatTextEndpoint(chatId), {
    headers: {
      Accept: 'application/json, text/plain',
    },
    signal,
  })

  if (!response.ok) {
    throw new Error(`Chat text request failed with status ${response.status}`)
  }

  const contentType = response.headers.get('content-type')

  if (contentType?.includes('application/json')) {
    const data: unknown = await response.json()

    if (
      typeof data === 'object' &&
      data !== null &&
      'generated_text' in data &&
      typeof data.generated_text === 'string'
    ) {
      return data.generated_text
    }

    if (
      typeof data === 'object' &&
      data !== null &&
      'text' in data &&
      typeof data.text === 'string'
    ) {
      return data.text
    }

    if (
      typeof data === 'object' &&
      data !== null &&
      'content' in data &&
      typeof data.content === 'string'
    ) {
      return data.content
    }

    throw new Error('Chat text response did not include generated text.')
  }

  return response.text()
}

interface ChatState {
  text: string
  transcriptionDraft:
    | {
        baseText: string
        sessionId: string
      }
    | null
  useWebSearch: boolean
  status: ChatStatus
  activeRequestAbortController: AbortController | null
  activeRequestId: string | null
  messages: MessageType[]
  streamingMessageId: string | null
  setText: (text: string) => void
  beginTranscriptionDraft: (sessionId: string) => void
  updateTranscriptionDraft: (sessionId: string, transcript: string) => void
  finishTranscriptionDraft: (sessionId: string, transcript: string) => void
  toggleWebSearch: () => void
  updateMessageContent: (messageId: string, newContent: string) => void
  updateMessageAudio: (messageId: string, audioUrl: string) => void
  completeAssistantResponse: (messageId: string) => void
  failAssistantResponse: (messageId: string) => void
  interruptAssistantResponse: () => Promise<void>
  fetchAssistantResponse: (
    messageId: string,
    history: ChatHistoryMessage[]
  ) => Promise<void>
  addUserMessage: (content: string) => void
  submitMessage: (content: string) => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  text: '',
  transcriptionDraft: null,
  useWebSearch: false,
  status: 'ready',
  activeRequestAbortController: null,
  activeRequestId: null,
  messages: initialMessages,
  streamingMessageId: null,
  setText: (text) => {
    set({ text, transcriptionDraft: null })
  },
  beginTranscriptionDraft: (sessionId) => {
    set((state) => ({
      transcriptionDraft: {
        baseText: state.text,
        sessionId,
      },
    }))
  },
  updateTranscriptionDraft: (sessionId, transcript) => {
    set((state) => {
      const draft =
        state.transcriptionDraft?.sessionId === sessionId
          ? state.transcriptionDraft
          : { baseText: state.text, sessionId }
      const cleanTranscript = transcript.trimStart()
      const separator =
        draft.baseText.trim() && cleanTranscript.trim() ? ' ' : ''

      return {
        text: `${draft.baseText}${separator}${cleanTranscript}`,
        transcriptionDraft: draft,
      }
    })
  },
  finishTranscriptionDraft: (sessionId, transcript) => {
    set((state) => {
      const draft =
        state.transcriptionDraft?.sessionId === sessionId
          ? state.transcriptionDraft
          : { baseText: state.text, sessionId }
      const cleanTranscript = transcript.trim()
      const separator =
        draft.baseText.trim() && cleanTranscript ? ' ' : ''

      return {
        text: `${draft.baseText}${separator}${cleanTranscript}`,
        transcriptionDraft: null,
      }
    })
  },
  toggleWebSearch: () => {
    set((state) => ({ useWebSearch: !state.useWebSearch }))
  },
  updateMessageContent: (messageId, newContent) => {
    set((state) => ({
      messages: state.messages.map((msg) => {
        if (msg.versions.some((v) => v.id === messageId)) {
          return {
            ...msg,
            versions: msg.versions.map((v) =>
              v.id === messageId
                ? { ...v, content: newContent, contentStatus: 'ready' }
                : v
            ),
          }
        }
        return msg
      }),
    }))
  },
  updateMessageAudio: (messageId, audioUrl) => {
    set((state) => ({
      messages: state.messages.map((msg) => {
        if (msg.versions.some((v) => v.id === messageId)) {
          return {
            ...msg,
            versions: msg.versions.map((v) =>
              v.id === messageId
                ? { ...v, audioPlaybackComplete: false, audioUrl }
                : v
            ),
          }
        }
        return msg
      }),
    }))
  },
  completeAssistantResponse: (messageId) => {
    set((state) => {
      if (state.streamingMessageId !== messageId) {
        return state
      }

      return {
        messages: state.messages.map((msg) => {
          if (msg.versions.some((v) => v.id === messageId)) {
            return {
              ...msg,
              versions: msg.versions.map((v) =>
                v.id === messageId
                  ? { ...v, audioPlaybackComplete: true }
                  : v
              ),
            }
          }
          return msg
        }),
        activeRequestAbortController: null,
        activeRequestId: null,
        status: 'ready',
        streamingMessageId: null,
      }
    })
  },
  failAssistantResponse: (messageId) => {
    set((state) => ({
      messages: state.messages.map((msg) => {
        if (msg.versions.some((v) => v.id === messageId)) {
          return {
            ...msg,
            versions: msg.versions.map((v) =>
              v.id === messageId
                ? {
                    ...v,
                    audioPlaybackComplete: true,
                    content: 'Failed to generate audio response.',
                    contentStatus: 'error',
                  }
                : v
            ),
          }
        }
        return msg
      }),
      activeRequestAbortController: null,
      activeRequestId: null,
      status: 'error',
      streamingMessageId: null,
    }))
  },
  interruptAssistantResponse: async () => {
    const { activeRequestAbortController, activeRequestId, streamingMessageId } =
      get()

    activeRequestAbortController?.abort()

    set((state) => ({
      activeRequestAbortController: null,
      activeRequestId: null,
      messages: streamingMessageId
        ? state.messages.map((msg) => {
            if (msg.versions.some((v) => v.id === streamingMessageId)) {
              return {
                ...msg,
                versions: msg.versions.map((v) =>
                  v.id === streamingMessageId
                    ? {
                        ...v,
                        audioPlaybackComplete: true,
                        audioUrl: undefined,
                        content: 'Generation interrupted.',
                        contentStatus: 'error',
                      }
                    : v
                ),
              }
            }
            return msg
          })
        : state.messages,
      status: 'ready',
      streamingMessageId: null,
    }))

    if (!activeRequestId) {
      return
    }

    try {
      await fetch(chatInterruptEndpoint(activeRequestId), { method: 'POST' })
    } catch {
      // The local abort has already restored the UI; backend cleanup is best effort.
    }
  },
  fetchAssistantResponse: async (messageId, history) => {
    const abortController = new AbortController()

    set({
      activeRequestAbortController: abortController,
      activeRequestId: null,
      status: 'streaming',
      streamingMessageId: messageId,
    })

    try {
      const chatId = await initializeChat(history, abortController.signal)
      set((state) =>
        state.streamingMessageId === messageId
          ? { activeRequestId: chatId }
          : state
      )
      get().updateMessageAudio(messageId, chatTtsStreamEndpoint(chatId))

      void fetchTextResponse(chatId, abortController.signal)
        .then((text) => {
          get().updateMessageContent(messageId, text)
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === 'AbortError') {
            return
          }
          get().failAssistantResponse(messageId)
        })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      get().failAssistantResponse(messageId)
    }
  },
  addUserMessage: (content) => {
    const userMessageId = createMessageId('user')
    const userMessage: MessageType = {
      from: 'user',
      key: userMessageId,
      versions: [
        {
          content,
          contentStatus: 'ready',
          id: userMessageId,
        },
      ],
    }

    const history = toChatHistory([...get().messages, userMessage])

    set((state) => ({ messages: [...state.messages, userMessage] }))

    const assistantMessageId = createMessageId('assistant')
    const assistantMessage: MessageType = {
      from: 'assistant',
      key: assistantMessageId,
      versions: [
        {
          audioPlaybackComplete: false,
          content: '',
          contentStatus: 'pending',
          id: assistantMessageId,
        },
      ],
    }

    set((state) => ({ messages: [...state.messages, assistantMessage] }))
    void get().fetchAssistantResponse(assistantMessageId, history)
  },
  submitMessage: (content) => {
    set({ status: 'submitted', text: '' })
    get().addUserMessage(content)
  },
}))
