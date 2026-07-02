'use client'

import type { ToolUIPart } from 'ai'
import { create } from 'zustand'

const apiEndpoint = import.meta.env.VITE_API_ENDPOINT ?? ''
const chatInitializeEndpoint = `${apiEndpoint}/initiate-request`
const chatTextEndpoint = (chatId: string) =>
  `${apiEndpoint}/requests/${chatId}/text`
const chatTtsStreamEndpoint = (chatId: string) =>
  `${apiEndpoint}/requests/${chatId}/audio`

export interface MessageType {
  key: string
  from: 'user' | 'assistant' | 'system'
  sources?: { href: string; title: string }[]
  versions: {
    id: string
    content: string
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

const initializeChat = async (messages: ChatHistoryMessage[]) => {
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

const fetchTextResponse = async (chatId: string) => {
  const response = await fetch(chatTextEndpoint(chatId), {
    headers: {
      Accept: 'application/json, text/plain',
    },
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
  useWebSearch: boolean
  status: ChatStatus
  messages: MessageType[]
  streamingMessageId: string | null
  setText: (text: string) => void
  appendTranscription: (transcript: string) => void
  toggleWebSearch: () => void
  updateMessageContent: (messageId: string, newContent: string) => void
  updateMessageAudio: (messageId: string, audioUrl: string) => void
  completeAssistantResponse: (messageId: string) => void
  failAssistantResponse: (messageId: string) => void
  fetchAssistantResponse: (
    messageId: string,
    history: ChatHistoryMessage[]
  ) => Promise<void>
  addUserMessage: (content: string) => void
  submitMessage: (content: string) => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  text: '',
  useWebSearch: false,
  status: 'ready',
  messages: initialMessages,
  streamingMessageId: null,
  setText: (text) => {
    set({ text })
  },
  appendTranscription: (transcript) => {
    set((state) => ({
      text: state.text ? `${state.text} ${transcript}` : transcript,
    }))
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
              v.id === messageId ? { ...v, content: newContent } : v
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
              v.id === messageId ? { ...v, audioUrl } : v
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

      return { status: 'ready', streamingMessageId: null }
    })
  },
  failAssistantResponse: (messageId) => {
    get().updateMessageContent(messageId, 'Failed to generate audio response.')
    set({ status: 'error', streamingMessageId: null })
  },
  fetchAssistantResponse: async (messageId, history) => {
    set({ status: 'streaming', streamingMessageId: messageId })

    try {
      const chatId = await initializeChat(history)
      get().updateMessageAudio(messageId, chatTtsStreamEndpoint(chatId))

      void fetchTextResponse(chatId)
        .then((text) => {
          get().updateMessageContent(messageId, text)
        })
        .catch(() => {
          get().failAssistantResponse(messageId)
        })
    } catch {
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
          content: 'Generating audio...',
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
