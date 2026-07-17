'use client'

import type { ToolUIPart } from 'ai'
import { create } from 'zustand'
import {
  createChatHistory,
  type StoredChatMessage,
  updateChatHistory,
} from '@/lib/chat-history'
import { usePersonasStore } from '@/src/state/personas'
import { usePreferencesStore } from '@/src/state/preferences'

const apiEndpoint = import.meta.env.VITE_API_ENDPOINT ?? "http://127.0.0.1:8000" // default-value local API, otherwise try to reach URL specified in .env
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

const createMessageId = (prefix: string) =>
  `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`

const toChatHistory = (messages: MessageType[]): StoredChatMessage[] =>
  messages.flatMap((message) =>
    message.versions
      .map((version) => ({
        content: version.content.trim(),
        role: message.from,
      }))
      .filter((message) => message.content)
  )

type ChatRequestParameters = {
  cloneVoice: boolean
  maxNewTokens: number
  refAudio?: string
  repeatPenalty: 1 | 1.1 | 1.2
  seed: number | null
  temperature: number
}

const initializeChat = async (
  messages: StoredChatMessage[],
  parameters: ChatRequestParameters,
  signal?: AbortSignal
) => {
  const response = await fetch(chatInitializeEndpoint, {
    body: JSON.stringify({
      clone_voice: parameters.cloneVoice,
      max_tokens: parameters.maxNewTokens,
      messages,
      ref_audio:
        parameters.cloneVoice && parameters.refAudio
          ? parameters.refAudio
          : null,
      repeat_penalty: parameters.repeatPenalty,
      response_format: 'pcm',
      seed: parameters.seed,
      stream_audio: true,
      temperature: parameters.temperature,
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

const toMessageTypes = (conversation: StoredChatMessage[]): MessageType[] =>
  conversation.map((message) => {
    const messageId = createMessageId(message.role)

    return {
      from: message.role,
      key: messageId,
      versions: [
        {
          audioPlaybackComplete: true,
          content: message.content,
          contentStatus: 'ready',
          id: messageId,
        },
      ],
    }
  })

const chatSaveChains = new Map<string, Promise<string>>()

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
  activeHistoryId: string | null
  activeChatKey: string
  activePersonaId: string | null
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
    history: StoredChatMessage[]
  ) => Promise<void>
  persistConversation: (conversation: StoredChatMessage[]) => void
  startNewChat: () => void
  clearDeletedChat: (recordId: string) => void
  loadChat: (
    recordId: string,
    conversation: StoredChatMessage[],
    personaId: string | null
  ) => void
  selectPersonaForNewChat: (personaId: string) => void
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
  activeHistoryId: null,
  activeChatKey: createMessageId('chat'),
  activePersonaId: usePersonasStore.getState().selectedPersonaId,
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
    get().persistConversation(toChatHistory(get().messages))
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
    if (get().streamingMessageId !== messageId) {
      return
    }

    set((state) => {
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
    get().persistConversation(toChatHistory(get().messages))
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
    get().persistConversation(toChatHistory(get().messages))
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
    get().persistConversation(toChatHistory(get().messages))

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
      try {
        await usePersonasStore.getState().ensurePersonasLoaded()
      } catch {
        // Persona loading should not prevent the user from chatting.
      }

      const personasState = usePersonasStore.getState()
      const personaId = get().activePersonaId ?? personasState.selectedPersonaId
      const persona = personasState.personas.find(
        (candidate) => candidate.id === personaId
      )
      const requestHistory: StoredChatMessage[] = persona?.instructionPrompt
        ? [
            { role: 'system', content: persona.instructionPrompt },
            ...history,
          ]
        : history
      const parameters =
        usePreferencesStore.getState().generationParameters

      if (get().streamingMessageId === messageId && personaId) {
        set({ activePersonaId: personaId })
      }

      const chatId = await initializeChat(
        requestHistory,
        {
          cloneVoice: parameters.cloneVoice,
          maxNewTokens: parameters.maxNewTokens,
          refAudio: persona?.audioSampleUrl ?? undefined,
          repeatPenalty: parameters.repeatPenalty,
          seed: parameters.seed,
          temperature: parameters.temperature,
        },
        abortController.signal
      )
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
  persistConversation: (conversation) => {
    if (conversation.length === 0) {
      return
    }

    const { activeChatKey, activeHistoryId } = get()
    const previousSave = chatSaveChains.get(activeChatKey)
    let nextSave: Promise<string>

    if (previousSave) {
      nextSave = previousSave.then(async (recordId) => {
        await updateChatHistory(recordId, conversation)
        return recordId
      })
    } else if (activeHistoryId) {
      nextSave = updateChatHistory(activeHistoryId, conversation).then(
        () => activeHistoryId
      )
    } else {
      nextSave = (async () => {
        let personaId = get().activePersonaId

        if (!personaId) {
          try {
            await usePersonasStore.getState().ensurePersonasLoaded()
            personaId = usePersonasStore.getState().selectedPersonaId
          } catch {
            // Save the chat without a persona if PocketBase persona loading fails.
          }
        }

        if (get().activeChatKey === activeChatKey && personaId) {
          set({ activePersonaId: personaId })
        }

        const record = await createChatHistory(conversation, personaId)
        return record.id
      })()
    }

    chatSaveChains.set(activeChatKey, nextSave)
    void nextSave
      .then((recordId) => {
        if (get().activeChatKey === activeChatKey) {
          set({ activeHistoryId: recordId })
        }
      })
      .catch(() => {
        if (chatSaveChains.get(activeChatKey) === nextSave) {
          chatSaveChains.delete(activeChatKey)
        }
      })
  },
  startNewChat: () => {
    void get().interruptAssistantResponse()
    set({
      activeChatKey: createMessageId('chat'),
      activeHistoryId: null,
      activePersonaId: usePersonasStore.getState().selectedPersonaId,
      messages: [],
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
    })
  },
  clearDeletedChat: (recordId) => {
    const {
      activeHistoryId,
      activeRequestAbortController,
      activeRequestId,
    } = get()

    if (activeHistoryId !== recordId) {
      return
    }

    activeRequestAbortController?.abort()
    set({
      activeChatKey: createMessageId('chat'),
      activeHistoryId: null,
      activePersonaId: usePersonasStore.getState().selectedPersonaId,
      activeRequestAbortController: null,
      activeRequestId: null,
      messages: [],
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
    })

    if (activeRequestId) {
      void fetch(chatInterruptEndpoint(activeRequestId), { method: 'POST' }).catch(
        () => {
          // The deleted chat is already cleared locally; cleanup is best effort.
        }
      )
    }
  },
  loadChat: (recordId, conversation, personaId) => {
    void get().interruptAssistantResponse()
    const restoredPersonaId =
      personaId ?? usePersonasStore.getState().selectedPersonaId

    if (restoredPersonaId) {
      usePersonasStore.getState().selectPersona(restoredPersonaId)
    }

    set({
      activeChatKey: `history-${recordId}`,
      activeHistoryId: recordId,
      activePersonaId: restoredPersonaId,
      messages: toMessageTypes(conversation),
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
    })
  },
  selectPersonaForNewChat: (personaId) => {
    set((state) =>
      state.messages.length === 0 && !state.activeHistoryId
        ? { activePersonaId: personaId }
        : state
    )
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
    get().persistConversation(history)
    void get().fetchAssistantResponse(assistantMessageId, history)
  },
  submitMessage: (content) => {
    set({ status: 'submitted', text: '' })
    get().addUserMessage(content)
  },
}))
