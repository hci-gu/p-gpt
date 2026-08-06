'use client'

import type { ToolUIPart } from 'ai'
import { create } from 'zustand'
import { getApiErrorMessage } from '@/lib/api-error'
import {
  createChatHistory,
  type StoredChatMessage,
  updateChatHistory,
} from '@/lib/chat-history'
import { usePersonasStore } from '@/src/state/personas'
import {
  omnivoiceNumStepsFromLevel,
  usePreferencesStore,
} from '@/src/state/preferences'

const apiEndpoint = import.meta.env.VITE_API_ENDPOINT ?? "http://127.0.0.1:8000" // default-value local API, otherwise try to reach URL specified in .env
const requestApiEndpoint = (streaming: boolean) =>
  streaming ? `${apiEndpoint}/pseudo-stream` : apiEndpoint
const chatInitializeEndpoint = (streaming: boolean) =>
  `${requestApiEndpoint(streaming)}/initiate-request`
const chatTextEndpoint = (chatId: string, streaming: boolean) =>
  `${requestApiEndpoint(streaming)}/requests/${chatId}/text`
const chatTtsStreamEndpoint = (chatId: string, streaming: boolean) =>
  `${requestApiEndpoint(streaming)}/requests/${chatId}/audio`
const chatInterruptEndpoint = (chatId: string, streaming: boolean) =>
  `${requestApiEndpoint(streaming)}/requests/${chatId}/interrupt`
const omniInferenceEndpoint = `${apiEndpoint}/omni/infer`

export interface MessageType {
  key: string
  from: 'user' | 'assistant' | 'system'
  sources?: { href: string; title: string }[]
  versions: {
    id: string
    content: string
    contentStatus?: 'pending' | 'ready' | 'error'
    audioError?: string
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
  'Hello!',
  'Who are you?',
  'Breifly describe yourself',
  'Welcome, how can I help you today?',
]

export type ChatStatus = 'submitted' | 'streaming' | 'ready' | 'error'

const createMessageId = (prefix: string) =>
  `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`

const toChatHistory = (messages: MessageType[]): StoredChatMessage[] =>
  messages.flatMap((message) =>
    message.versions
      .filter((version) => version.contentStatus !== 'error')
      .map((version) => ({
        content: version.content.trim(),
        role: message.from,
      }))
      .filter((message) => message.content)
  )

type ChatRequestParameters = {
  cloneVoice: boolean
  maxNewTokens: number
  model: string | null
  numSteps: number
  refAudio?: string
  repeatPenalty: 1 | 1.1 | 1.2
  seed: number | null
  streaming: boolean
  temperature: number
}

const initializeChat = async (
  messages: StoredChatMessage[],
  personaId: string,
  personaName: string,
  instructionPrompt: string,
  parameters: ChatRequestParameters,
  signal?: AbortSignal
) => {
  const response = await fetch(chatInitializeEndpoint(parameters.streaming), {
    body: JSON.stringify({
      clone_voice: parameters.cloneVoice,
      instruction_prompt: instructionPrompt,
      max_tokens: parameters.maxNewTokens,
      messages,
      ...(parameters.model ? { model: parameters.model } : {}),
      num_step: parameters.numSteps,
      persona_id: personaId,
      persona_name: personaName,
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
    throw new Error(await getApiErrorMessage(response, 'Chat initialization failed'))
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

const fetchTextResponse = async (
  chatId: string,
  streaming: boolean,
  signal?: AbortSignal
) => {
  const response = await fetch(chatTextEndpoint(chatId, streaming), {
    headers: {
      Accept: 'application/json, text/plain',
    },
    signal,
  })

  if (!response.ok) {
    throw new Error(await getApiErrorMessage(response, 'Text generation failed'))
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

type OmniResponseParameters = {
  audio: Float32Array
  inputText: string
  sampleRate: number
}

type QueuedOmniAudio = OmniResponseParameters

const encodePcm16Wav = (audio: Float32Array, sampleRate: number) => {
  const buffer = new ArrayBuffer(44 + audio.length * 2)
  const view = new DataView(buffer)
  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + audio.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, audio.length * 2, true)

  for (let index = 0; index < audio.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, audio[index] ?? 0))
    view.setInt16(44 + index * 2, sample * 32_767, true)
  }

  return new Uint8Array(buffer)
}

const bytesToBase64 = (bytes: Uint8Array) => {
  let binary = ''
  const chunkSize = 0x8000

  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(
      ...bytes.subarray(offset, Math.min(bytes.length, offset + chunkSize))
    )
  }

  return btoa(binary)
}

const fetchOmniResponse = async (
  messages: StoredChatMessage[],
  personaId: string,
  personaName: string,
  instructionPrompt: string,
  inputText: string,
  audio: Float32Array,
  sampleRate: number,
  signal: AbortSignal
) => {
  const response = await fetch(omniInferenceEndpoint, {
    body: JSON.stringify({
      audio_base64: bytesToBase64(encodePcm16Wav(audio, sampleRate)),
      audio_sample_rate: sampleRate,
      input_text: inputText,
      instruction_prompt: instructionPrompt,
      max_tokens: 256,
      messages,
      persona_id: personaId,
      persona_name: personaName,
      temperature: 0.7,
      top_p: 0.95,
    }),
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    method: 'POST',
    signal,
  })

  if (!response.ok) {
    throw new Error(await getApiErrorMessage(response, 'Omni model failed'))
  }

  const data: unknown = await response.json()
  if (
    typeof data !== 'object' ||
    data === null ||
    !('generated_text' in data) ||
    typeof data.generated_text !== 'string'
  ) {
    throw new Error('Omni model response did not include generated text.')
  }

  return {
    audioBase64:
      'audio_base64' in data && typeof data.audio_base64 === 'string'
        ? data.audio_base64
        : null,
    text: data.generated_text,
  }
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
  activeRequestStreaming: boolean
  messages: MessageType[]
  streamingMessageId: string | null
  activeHistoryId: string | null
  activeChatKey: string
  activePersonaId: string | null
  queuedOmniAudio: QueuedOmniAudio | null
  setText: (text: string) => void
  beginTranscriptionDraft: (sessionId: string) => void
  updateTranscriptionDraft: (sessionId: string, transcript: string) => void
  finishTranscriptionDraft: (sessionId: string, transcript: string) => void
  toggleWebSearch: () => void
  updateMessageContent: (messageId: string, newContent: string) => void
  updateMessageAudio: (messageId: string, audioUrl: string) => void
  completeAssistantResponse: (messageId: string) => void
  failAssistantResponse: (
    messageId: string,
    errorMessage: string,
    preserveContent?: boolean
  ) => void
  interruptAssistantResponse: () => Promise<void>
  fetchAssistantResponse: (
    messageId: string,
    history: StoredChatMessage[]
  ) => Promise<void>
  fetchOmniAssistantResponse: (
    messageId: string,
    history: StoredChatMessage[],
    parameters: OmniResponseParameters
  ) => Promise<void>
  persistConversation: (conversation: StoredChatMessage[]) => void
  resetForAuthChange: () => void
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
  submitOmniAudio: (
    content: string,
    audio: Float32Array,
    sampleRate: number
  ) => void
}

export const useChatStore = create<ChatState>((set, get) => {
  const startOmniAudio = (
    content: string,
    audio: Float32Array,
    sampleRate: number
  ) => {
    const userMessageId = createMessageId('user')
    const userMessage: MessageType = {
      from: 'user',
      key: userMessageId,
      versions: [
        {
          content: content.trim() || 'Voice input',
          contentStatus: 'ready',
          id: userMessageId,
        },
      ],
    }
    const history = toChatHistory([...get().messages, userMessage])
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

    set((state) => ({
      messages: [...state.messages, userMessage, assistantMessage],
      status: 'submitted',
    }))
    get().persistConversation(history)
    void get().fetchOmniAssistantResponse(assistantMessageId, history, {
      audio,
      inputText: content,
      sampleRate,
    })
  }

  const drainQueuedOmniAudio = () => {
    const state = get()
    if (
      !state.queuedOmniAudio ||
      state.status === 'submitted' ||
      state.status === 'streaming'
    ) {
      return
    }

    const queuedAudio = state.queuedOmniAudio
    set({ queuedOmniAudio: null })
    startOmniAudio(
      queuedAudio.inputText,
      queuedAudio.audio,
      queuedAudio.sampleRate
    )
  }

  return {
  text: '',
  transcriptionDraft: null,
  useWebSearch: false,
  status: 'ready',
  activeRequestAbortController: null,
  activeRequestId: null,
  activeRequestStreaming: false,
  messages: initialMessages,
  streamingMessageId: null,
  activeHistoryId: null,
  activeChatKey: createMessageId('chat'),
  activePersonaId: usePersonasStore.getState().selectedPersonaId,
  queuedOmniAudio: null,
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
                ? {
                    ...v,
                    content: v.audioError
                      ? `${newContent}\n\n${v.audioError}`
                      : newContent,
                    contentStatus: v.audioError ? 'error' : 'ready',
                  }
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
        activeRequestStreaming: false,
        status: 'ready',
        streamingMessageId: null,
      }
    })
    get().persistConversation(toChatHistory(get().messages))
    drainQueuedOmniAudio()
  },
  failAssistantResponse: (messageId, errorMessage, preserveContent = false) => {
    set((state) => ({
      messages: state.messages.map((msg) => {
        if (msg.versions.some((v) => v.id === messageId)) {
          return {
            ...msg,
            versions: msg.versions.map((v) =>
              v.id === messageId
                ? {
                    ...v,
                    audioError: preserveContent ? errorMessage : undefined,
                    audioPlaybackComplete: true,
                    audioUrl: undefined,
                    content:
                      preserveContent && v.content.trim()
                        ? `${v.content}\n\n${errorMessage}`
                        : errorMessage,
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
      activeRequestStreaming: false,
      status: 'error',
      streamingMessageId: null,
    }))
    get().persistConversation(toChatHistory(get().messages))
    drainQueuedOmniAudio()
  },
  interruptAssistantResponse: async () => {
    const {
      activeRequestAbortController,
      activeRequestId,
      activeRequestStreaming,
      streamingMessageId,
    } = get()

    activeRequestAbortController?.abort()

    set((state) => ({
      activeRequestAbortController: null,
      activeRequestId: null,
      activeRequestStreaming: false,
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
      drainQueuedOmniAudio()
      return
    }

    try {
      await fetch(chatInterruptEndpoint(activeRequestId, activeRequestStreaming), {
        method: 'POST',
      })
    } catch {
      // The local abort has already restored the UI; backend cleanup is best effort.
    }
    drainQueuedOmniAudio()
  },
  fetchAssistantResponse: async (messageId, history) => {
    const abortController = new AbortController()

    set({
      activeRequestAbortController: abortController,
      activeRequestId: null,
      activeRequestStreaming: false,
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
      if (!persona) {
        throw new Error('A valid persona must be selected before chatting.')
      }
      const requestHistory = history.filter(
        (message) => message.role !== 'system'
      )
      const parameters =
        usePreferencesStore.getState().generationParameters

      if (get().streamingMessageId === messageId && personaId) {
        set({ activePersonaId: personaId })
      }

      const chatId = await initializeChat(
        requestHistory,
        persona.id,
        persona.name,
        persona.instructionPrompt,
        {
          cloneVoice: parameters.cloneVoice,
          maxNewTokens: parameters.maxNewTokens,
          model: parameters.model,
          numSteps: omnivoiceNumStepsFromLevel(parameters.ttsStepLevel),
          refAudio: persona?.audioSampleUrl ?? undefined,
          repeatPenalty: parameters.repeatPenalty,
          seed: parameters.seed,
          streaming: parameters.streaming,
          temperature: parameters.temperature,
        },
        abortController.signal
      )
      set((state) =>
        state.streamingMessageId === messageId
          ? {
              activeRequestId: chatId,
              activeRequestStreaming: parameters.streaming,
            }
          : state
      )
      get().updateMessageAudio(
        messageId,
        chatTtsStreamEndpoint(chatId, parameters.streaming)
      )

      void fetchTextResponse(chatId, parameters.streaming, abortController.signal)
        .then((text) => {
          get().updateMessageContent(messageId, text)
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === 'AbortError') {
            return
          }
          get().failAssistantResponse(
            messageId,
            error instanceof Error
              ? error.message
              : 'Text generation failed: unknown error.'
          )
        })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      get().failAssistantResponse(
        messageId,
        error instanceof Error
          ? error.message
          : 'Chat initialization failed: unknown error.'
      )
    }
  },
  fetchOmniAssistantResponse: async (messageId, history, parameters) => {
    const abortController = new AbortController()

    set({
      activeRequestAbortController: abortController,
      activeRequestId: null,
      activeRequestStreaming: false,
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
      if (!persona) {
        throw new Error('A valid persona must be selected before chatting.')
      }

      const response = await fetchOmniResponse(
        history,
        persona.id,
        persona.name,
        persona.instructionPrompt,
        parameters.inputText,
        parameters.audio,
        parameters.sampleRate,
        abortController.signal
      )

      if (get().streamingMessageId !== messageId) {
        return
      }

      get().updateMessageContent(messageId, response.text)
      if (response.audioBase64) {
        get().updateMessageAudio(
          messageId,
          `data:audio/pcm;base64,${response.audioBase64}`
        )
      } else {
        get().completeAssistantResponse(messageId)
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      get().failAssistantResponse(
        messageId,
        error instanceof Error
          ? error.message
          : 'Omni model failed: unknown error.'
      )
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
  resetForAuthChange: () => {
    get().activeRequestAbortController?.abort()
    chatSaveChains.clear()
    set({
      activeChatKey: createMessageId('chat'),
      activeHistoryId: null,
      activePersonaId: null,
      activeRequestAbortController: null,
      activeRequestId: null,
      activeRequestStreaming: false,
      messages: [],
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
      queuedOmniAudio: null,
      useWebSearch: false,
    })
  },
  startNewChat: () => {
    void get().interruptAssistantResponse()
    set({
      activeChatKey: createMessageId('chat'),
      activeHistoryId: null,
      activePersonaId: usePersonasStore.getState().selectedPersonaId,
      activeRequestAbortController: null,
      activeRequestId: null,
      activeRequestStreaming: false,
      messages: [],
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
      queuedOmniAudio: null,
    })
  },
  clearDeletedChat: (recordId) => {
    const {
      activeHistoryId,
      activeRequestAbortController,
      activeRequestId,
      activeRequestStreaming,
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
      activeRequestStreaming: false,
      messages: [],
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
      queuedOmniAudio: null,
    })

    if (activeRequestId) {
      void fetch(chatInterruptEndpoint(activeRequestId, activeRequestStreaming), {
        method: 'POST',
      }).catch(() => {
        // The deleted chat is already cleared locally; cleanup is best effort.
      })
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
      activeRequestAbortController: null,
      activeRequestId: null,
      activeRequestStreaming: false,
      messages: toMessageTypes(conversation),
      status: 'ready',
      streamingMessageId: null,
      text: '',
      transcriptionDraft: null,
      queuedOmniAudio: null,
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
  submitOmniAudio: (content, audio, sampleRate) => {
    const state = get()
    if (state.status === 'submitted' || state.status === 'streaming') {
      if (!state.queuedOmniAudio) {
        set({
          queuedOmniAudio: {
            audio,
            inputText: content,
            sampleRate,
          },
        })
      }
      return
    }

    startOmniAudio(content, audio, sampleRate)
  },
  }
})
