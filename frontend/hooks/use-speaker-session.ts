import { useCallback, useEffect, useRef, useState } from 'react'
import type { PersonaRecord } from '@/lib/personas'
import {
  float32ToPcm16,
  type SpeakerAudioConsumer,
} from '@/src/lib/speaker-audio'
import { SpeakerPcmPlayback } from '@/src/lib/speaker-playback'
import {
  createSpeakerEvent,
  getSpeakerSocketUrl,
  type SpeakerServerEvent,
  type SpeakerStatus,
  speakerWebSocketProtocol,
} from '@/src/lib/speaker-protocol'
import { useChatStore } from '@/src/state/chat'
import {
  omnivoiceNumStepsFromLevel,
  usePreferencesStore,
} from '@/src/state/preferences'
import type { SpeechLanguage } from '@/src/lib/speech-language'

const apiEndpoint =
  import.meta.env.VITE_API_ENDPOINT ?? 'http://127.0.0.1:8000'
const maximumBufferedBytes = 256 * 1024

type PipelinePhase = 'idle' | 'capturing' | 'grace' | 'responding'

export const useSpeakerSession = ({
  enabled,
  persona,
  volume,
}: {
  enabled: boolean
  persona: PersonaRecord | undefined
  volume: number
}) => {
  const generationParameters = usePreferencesStore(
    (state) => state.generationParameters
  )
  const setGenerationParameter = usePreferencesStore(
    (state) => state.setGenerationParameter
  )
  const speechLanguage = generationParameters.speechLanguage
  const generationConfigurationKey = JSON.stringify({
    cloneVoice: generationParameters.cloneVoice,
    maxNewTokens: generationParameters.maxNewTokens,
    model: generationParameters.model,
    repeatPenalty: generationParameters.repeatPenalty,
    seed: generationParameters.seed,
    temperature: generationParameters.temperature,
    ttsStepLevel: generationParameters.ttsStepLevel,
  })
  const activeChatKey = useChatStore((state) => state.activeChatKey)
  const commitUser = useChatStore(
    (state) => state.commitSpeakerUserMessage
  )
  const commitAssistant = useChatStore(
    (state) => state.commitSpeakerAssistantMessage
  )
  const [status, setStatus] = useState<SpeakerStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLanguageUpdating, setIsLanguageUpdating] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const [transcriptRenderSequence, setTranscriptRenderSequence] = useState(0)
  const [latestUserTranscript, setLatestUserTranscript] = useState<string | null>(
    null
  )
  const [latestAssistantTranscript, setLatestAssistantTranscript] = useState<
    string | null
  >(null)

  const socketRef = useRef<WebSocket | null>(null)
  const sessionReadyRef = useRef(false)
  const phaseRef = useRef<PipelinePhase>('idle')
  const turnIdRef = useRef<string | null>(null)
  const turnRevisionRef = useRef(0)
  const generationRef = useRef<number | null>(null)
  const binaryGenerationRef = useRef<number | null>(null)
  const pendingInterruptedGenerationRef = useRef<number | null>(null)
  const playedTextRef = useRef(new Map<number, string[]>())
  const committedGenerationsRef = useRef(new Set<number>())
  const sessionChatKeyRef = useRef(activeChatKey)
  const desiredLanguageRef = useRef<SpeechLanguage>(speechLanguage)
  const acknowledgedLanguageRef = useRef<SpeechLanguage>(speechLanguage)
  const pendingLanguageRef = useRef<SpeechLanguage | null>(null)
  const languageUpdatingRef = useRef(false)
  const generationParametersRef = useRef(generationParameters)
  const pendingTranscriptRenderRef = useRef<{
    generation: number
    receivedAtMilliseconds: number
  } | null>(null)
  const sendEventRef = useRef<(type: string, values?: Record<string, unknown>) => void>(
    () => undefined
  )
  const commitAssistantRef = useRef(commitAssistant)
  commitAssistantRef.current = commitAssistant
  desiredLanguageRef.current = speechLanguage
  generationParametersRef.current = generationParameters

  const requestLanguageUpdateRef = useRef<() => void>(() => undefined)
  const requestLanguageUpdate = useCallback(() => {
    const desiredLanguage = desiredLanguageRef.current
    if (
      !sessionReadyRef.current ||
      phaseRef.current !== 'idle' ||
      pendingLanguageRef.current !== null ||
      desiredLanguage === acknowledgedLanguageRef.current
    ) {
      return
    }
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return
    }
    pendingLanguageRef.current = desiredLanguage
    languageUpdatingRef.current = true
    setIsLanguageUpdating(true)
    sendEventRef.current('session.update', {
      inputLanguage: desiredLanguage,
    })
    console.debug('[speaker-session] language update requested', {
      inputLanguage: desiredLanguage,
    })
  }, [])
  requestLanguageUpdateRef.current = requestLanguageUpdate

  useEffect(() => {
    if (enabled) {
      requestLanguageUpdate()
    }
  }, [enabled, requestLanguageUpdate, speechLanguage])

  const playerRef = useRef<SpeakerPcmPlayback | null>(null)
  if (!playerRef.current && typeof window !== 'undefined') {
    playerRef.current = new SpeakerPcmPlayback(volume, {
      onDiagnostic: (diagnostic) => {
        console.debug('[speaker-playback] diagnostic', diagnostic)
        sendEventRef.current('client.speaker_diagnostic', {
          activity: diagnostic.activity,
          chunkCount: diagnostic.chunkCount,
          minimumSchedulingLeadMilliseconds:
            diagnostic.minimumSchedulingLeadMilliseconds,
          phase: phaseRef.current,
          responseGeneration: diagnostic.generation,
          schedulingLeadMilliseconds:
            diagnostic.schedulingLeadMilliseconds,
          segmentId: diagnostic.segmentId,
          underrunCount: diagnostic.underrunCount,
        })
      },
      onLevelChange: setAudioLevel,
      onPlayingChange: (playing) => {
        setIsPlaying(playing)
        if (playing) {
          setStatus('speaking')
        } else if (phaseRef.current === 'responding') {
          setStatus('thinking')
        }
      },
      onResponseCompleted: (generation) => {
        sendEventRef.current('playback.response_completed', {
          responseGeneration: generation,
        })
      },
      onSegmentCompleted: (generation, segmentId, text) => {
        const played = playedTextRef.current.get(generation) ?? []
        played.push(text)
        playedTextRef.current.set(generation, played)
        sendEventRef.current('playback.segment_completed', {
          responseGeneration: generation,
          segmentId,
        })
      },
    })
  }

  useEffect(() => {
    playerRef.current?.setVolume(volume)
  }, [volume])

  useEffect(() => {
    const pending = pendingTranscriptRenderRef.current
    if (!pending || transcriptRenderSequence === 0) {
      return
    }
    pendingTranscriptRenderRef.current = null
    const receiveToRenderMilliseconds =
      performance.now() - pending.receivedAtMilliseconds
    console.debug('[speaker-session] transcript rendered', {
      generation: pending.generation,
      receiveToRenderMilliseconds,
    })
    sendEventRef.current('client.speaker_diagnostic', {
      activity: 'transcript_rendered',
      phase: phaseRef.current,
      receiveToRenderMilliseconds,
      responseGeneration: pending.generation,
    })
  }, [transcriptRenderSequence])

  const commitInterruptedLocally = useCallback((generation: number | null) => {
    if (generation === null || committedGenerationsRef.current.has(generation)) {
      return
    }
    if (
      useChatStore.getState().activeChatKey !== sessionChatKeyRef.current
    ) {
      console.debug(
        '[speaker-session] skipped interrupted commit after conversation switch',
        { generation }
      )
      return
    }
    const text = (playedTextRef.current.get(generation) ?? []).join(' ').trim()
    if (!text) {
      return
    }
    committedGenerationsRef.current.add(generation)
    setLatestAssistantTranscript(text)
    commitAssistantRef.current(text, 'interrupted')
  }, [])

  useEffect(() => {
    setLatestUserTranscript(null)
    setLatestAssistantTranscript(null)
    setError(null)
  }, [activeChatKey])

  useEffect(() => {
    if (!enabled) {
      setStatus('idle')
      setIsReady(false)
      setError(null)
      phaseRef.current = 'idle'
      sessionReadyRef.current = false
      pendingLanguageRef.current = null
      languageUpdatingRef.current = false
      setIsLanguageUpdating(false)
      turnIdRef.current = null
      generationRef.current = null
      playerRef.current?.clear()
      return undefined
    }
    if (!persona) {
      setStatus('error')
      setError('A Persona must be selected before speaker mode can connect.')
      setIsReady(false)
      return undefined
    }

    let disposed = false
    let intentionalClose = false
    let fatal = false
    let reconnectAttempts = 0
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    sessionChatKeyRef.current = activeChatKey

    const isCurrentConversation = () =>
      useChatStore.getState().activeChatKey === activeChatKey

    const sendEvent = (type: string, values: Record<string, unknown> = {}) => {
      const socket = socketRef.current
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return
      }
      socket.send(JSON.stringify(createSpeakerEvent(type, values)))
    }
    sendEventRef.current = sendEvent

    const finalizeServerResponse = (
      generation: number,
      text: string,
      interrupted: boolean
    ) => {
      if (!text.trim() || committedGenerationsRef.current.has(generation)) {
        return
      }
      committedGenerationsRef.current.add(generation)
      setLatestAssistantTranscript(text.trim())
      commitAssistant(text.trim(), interrupted ? 'interrupted' : undefined)
    }

    const handleServerEvent = (serverEvent: SpeakerServerEvent) => {
      if (!isCurrentConversation()) {
        console.debug(
          '[speaker-session] dropped event for the previous conversation',
          serverEvent.type
        )
        return
      }
      console.debug('[speaker-session] server event', {
        generation: serverEvent.responseGeneration,
        inputLanguage: serverEvent.inputLanguage,
        revision: serverEvent.turnRevision,
        segmentId: serverEvent.segmentId,
        turnId: serverEvent.turnId,
        type: serverEvent.type,
      })
      switch (serverEvent.type) {
        case 'session.ready':
          reconnectAttempts = 0
          setError(null)
          setIsReady(true)
          sessionReadyRef.current = true
          acknowledgedLanguageRef.current =
            serverEvent.inputLanguage ?? desiredLanguageRef.current
          pendingLanguageRef.current = null
          languageUpdatingRef.current = false
          setIsLanguageUpdating(false)
          setStatus('listening')
          phaseRef.current = 'idle'
          queueMicrotask(() => requestLanguageUpdateRef.current())
          return
        case 'session.updated':
          if (
            serverEvent.inputLanguage !== 'en' &&
            serverEvent.inputLanguage !== 'sv'
          ) {
            pendingLanguageRef.current = null
            languageUpdatingRef.current = false
            setIsLanguageUpdating(false)
            setGenerationParameter(
              'speechLanguage',
              acknowledgedLanguageRef.current
            )
            setError('The speaker server returned an invalid language update.')
            return
          }
          acknowledgedLanguageRef.current = serverEvent.inputLanguage
          pendingLanguageRef.current = null
          languageUpdatingRef.current = false
          setIsLanguageUpdating(false)
          setError(null)
          console.debug('[speaker-session] language update acknowledged', {
            inputLanguage: serverEvent.inputLanguage,
          })
          queueMicrotask(() => requestLanguageUpdateRef.current())
          return
        case 'input.transcription.committed':
          if (serverEvent.text) {
            const receivedAtMilliseconds = performance.now()
            if (serverEvent.responseGeneration !== undefined) {
              pendingTranscriptRenderRef.current = {
                generation: serverEvent.responseGeneration,
                receivedAtMilliseconds,
              }
            }
            console.debug('[speaker-session] transcript commit received', {
              generation: serverEvent.responseGeneration,
              receivedAtMilliseconds,
            })
            setLatestUserTranscript(serverEvent.text)
            setLatestAssistantTranscript(null)
            commitUser(serverEvent.text)
            if (serverEvent.responseGeneration !== undefined) {
              setTranscriptRenderSequence((value) => value + 1)
            }
          }
          phaseRef.current = 'responding'
          return
        case 'input.transcription.empty':
          phaseRef.current = 'idle'
          generationRef.current = null
          setStatus('listening')
          queueMicrotask(() => requestLanguageUpdateRef.current())
          return
        case 'response.started':
          if (typeof serverEvent.responseGeneration === 'number') {
            generationRef.current = serverEvent.responseGeneration
            playedTextRef.current.set(serverEvent.responseGeneration, [])
          }
          phaseRef.current = 'responding'
          setStatus('thinking')
          return
        case 'response.audio.segment_started':
          if (
            typeof serverEvent.responseGeneration === 'number' &&
            serverEvent.responseGeneration === generationRef.current &&
            serverEvent.segmentId &&
            typeof serverEvent.text === 'string'
          ) {
            binaryGenerationRef.current = serverEvent.responseGeneration
            playerRef.current?.beginSegment(
              serverEvent.responseGeneration,
              serverEvent.segmentId,
              serverEvent.text
            )
          }
          return
        case 'response.audio.segment_done':
          if (
            typeof serverEvent.responseGeneration === 'number' &&
            serverEvent.segmentId &&
            serverEvent.responseGeneration === binaryGenerationRef.current
          ) {
            playerRef.current?.endSegment(
              serverEvent.responseGeneration,
              serverEvent.segmentId
            )
            binaryGenerationRef.current = null
          }
          return
        case 'response.audio.done':
          if (
            typeof serverEvent.responseGeneration === 'number' &&
            serverEvent.responseGeneration === generationRef.current
          ) {
            playerRef.current?.endResponse(serverEvent.responseGeneration)
          }
          return
        case 'response.completed':
          if (
            typeof serverEvent.responseGeneration !== 'number' ||
            serverEvent.responseGeneration !== generationRef.current
          ) {
            return
          }
          if (typeof serverEvent.text === 'string') {
            finalizeServerResponse(serverEvent.responseGeneration, serverEvent.text, false)
          }
          phaseRef.current = 'idle'
          generationRef.current = null
          setStatus('listening')
          queueMicrotask(() => requestLanguageUpdateRef.current())
          return
        case 'response.cancelled':
          if (typeof serverEvent.responseGeneration !== 'number') {
            return
          }
          const cancelledCurrent =
            serverEvent.responseGeneration === generationRef.current
          const cancelledPending =
            serverEvent.responseGeneration ===
            pendingInterruptedGenerationRef.current
          if (!cancelledCurrent && !cancelledPending) {
            return
          }
          if (cancelledCurrent) {
            playerRef.current?.clear()
          }
          if (
            typeof serverEvent.responseGeneration === 'number' &&
            typeof serverEvent.text === 'string'
          ) {
            finalizeServerResponse(
              serverEvent.responseGeneration,
              serverEvent.text,
              true
            )
          }
          if (
            pendingInterruptedGenerationRef.current ===
            serverEvent.responseGeneration
          ) {
            pendingInterruptedGenerationRef.current = null
          }
          if (cancelledCurrent && phaseRef.current !== 'capturing') {
            phaseRef.current = 'idle'
            generationRef.current = null
            setStatus('listening')
            queueMicrotask(() => requestLanguageUpdateRef.current())
          }
          return
        case 'error':
          if (pendingLanguageRef.current !== null) {
            const acknowledgedLanguage = acknowledgedLanguageRef.current
            pendingLanguageRef.current = null
            languageUpdatingRef.current = false
            setIsLanguageUpdating(false)
            setGenerationParameter('speechLanguage', acknowledgedLanguage)
            setError(
              serverEvent.message ?? 'Could not change speaker input language.'
            )
            setStatus('listening')
            return
          }
          setError(serverEvent.message ?? 'Speaker mode encountered an error.')
          setStatus('error')
          if (serverEvent.fatal) {
            fatal = true
            setIsReady(false)
          }
          return
        default:
          return
      }
    }

    const connect = () => {
      if (disposed || fatal) {
        return
      }
      playedTextRef.current.clear()
      committedGenerationsRef.current.clear()
      pendingInterruptedGenerationRef.current = null
      generationRef.current = null
      binaryGenerationRef.current = null
      setStatus(reconnectAttempts ? 'reconnecting' : 'connecting')
      sessionReadyRef.current = false
      const socket = new WebSocket(
        getSpeakerSocketUrl(apiEndpoint),
        speakerWebSocketProtocol
      )
      socket.binaryType = 'arraybuffer'
      socketRef.current = socket

      socket.addEventListener('open', () => {
        console.debug('[speaker-session] socket opened', { activeChatKey })
        const history = useChatStore.getState().getConversationHistory()
        const currentGenerationParameters = generationParametersRef.current
        socket.send(
          JSON.stringify(
            createSpeakerEvent('session.configure', {
              protocolVersion: 1,
              personaId: persona.id,
              personaName: persona.name,
              instructionPrompt: persona.instructionPrompt,
              history,
              inputLanguage: desiredLanguageRef.current,
              generation: {
                model: currentGenerationParameters.model,
                temperature: currentGenerationParameters.temperature,
                repeatPenalty: currentGenerationParameters.repeatPenalty,
                seed: currentGenerationParameters.seed,
                maxTokens: currentGenerationParameters.maxNewTokens,
                cloneVoice: currentGenerationParameters.cloneVoice,
                refAudio: persona.audioSampleUrl ?? null,
                numStep: omnivoiceNumStepsFromLevel(
                  currentGenerationParameters.ttsStepLevel
                ),
                speed: 0.8,
              },
              inputAudio: {
                encoding: 'pcm_s16le',
                sampleRate: 16_000,
                channels: 1,
                frameSamples: 512,
              },
              outputAudio: {
                encoding: 'pcm_s16le',
                sampleRate: 24_000,
                channels: 1,
              },
            })
          )
        )
      })

      socket.addEventListener('message', (message) => {
        if (message.data instanceof ArrayBuffer) {
          const generation = binaryGenerationRef.current
          if (generation !== null && generation === generationRef.current) {
            void playerRef.current
              ?.pushPcm16(generation, message.data)
              .catch((caughtError: unknown) => {
                setError(
                  caughtError instanceof Error
                    ? caughtError.message
                    : 'Speaker audio playback failed.'
                )
                setStatus('error')
              })
          }
          return
        }
        try {
          handleServerEvent(JSON.parse(String(message.data)) as SpeakerServerEvent)
        } catch {
          setError('The speaker server returned an invalid event.')
          setStatus('error')
        }
      })

      socket.addEventListener('error', () => {
        console.debug('[speaker-session] socket error', { activeChatKey })
        setError('The speaker connection failed.')
      })

      socket.addEventListener('close', () => {
        console.debug('[speaker-session] socket closed', {
          activeChatKey,
          phase: phaseRef.current,
        })
        if (socketRef.current === socket) {
          socketRef.current = null
        }
        setIsReady(false)
        sessionReadyRef.current = false
        if (pendingLanguageRef.current !== null) {
          setGenerationParameter(
            'speechLanguage',
            acknowledgedLanguageRef.current
          )
        }
        pendingLanguageRef.current = null
        languageUpdatingRef.current = false
        setIsLanguageUpdating(false)
        playerRef.current?.clear()
        commitInterruptedLocally(generationRef.current)
        commitInterruptedLocally(pendingInterruptedGenerationRef.current)
        pendingInterruptedGenerationRef.current = null
        generationRef.current = null
        binaryGenerationRef.current = null
        phaseRef.current = 'idle'
        if (disposed || intentionalClose || fatal) {
          return
        }
        reconnectAttempts += 1
        setStatus('reconnecting')
        const delay = Math.min(4_000, 500 * 2 ** (reconnectAttempts - 1))
        reconnectTimer = setTimeout(connect, delay)
      })
    }

    connect()
    return () => {
      disposed = true
      intentionalClose = true
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
      }
      commitInterruptedLocally(generationRef.current)
      commitInterruptedLocally(pendingInterruptedGenerationRef.current)
      pendingInterruptedGenerationRef.current = null
      sendEvent('response.cancel', {
        responseGeneration: generationRef.current,
      })
      socketRef.current?.close(1000, 'Speaker mode closed')
      socketRef.current = null
      playerRef.current?.clear()
      setIsReady(false)
    }
  }, [
    activeChatKey,
    commitAssistant,
    commitInterruptedLocally,
    commitUser,
    enabled,
    generationConfigurationKey,
    persona,
    setGenerationParameter,
  ])

  const onAudioEvent = useCallback<SpeakerAudioConsumer>((audioEvent) => {
    const sendEvent = sendEventRef.current
    const socket = socketRef.current

    if (audioEvent.type === 'vad-diagnostic') {
      sendEvent('client.vad_diagnostic', {
        activity: audioEvent.activity,
        captureEpoch: audioEvent.captureEpoch,
        detail: audioEvent.detail,
        detectionProfile: audioEvent.detectionProfile,
        pendingFrameCount: audioEvent.pendingFrameCount,
        phase: phaseRef.current,
        processingAverageMilliseconds:
          audioEvent.processingAverageMilliseconds,
        processingMaximumMilliseconds:
          audioEvent.processingMaximumMilliseconds,
        probabilityAverage: audioEvent.probabilityAverage,
        probabilityMax: audioEvent.probabilityMax,
        probabilityMin: audioEvent.probabilityMin,
        queueDelayAverageMilliseconds:
          audioEvent.queueDelayAverageMilliseconds,
        queueDelayMaximumMilliseconds:
          audioEvent.queueDelayMaximumMilliseconds,
        recoveryCount: audioEvent.recoveryCount,
        sampleCount: audioEvent.sampleCount,
      })
      return
    }

    if (languageUpdatingRef.current) {
      console.debug('[speaker-session] dropped microphone event during language update', {
        type: audioEvent.type,
      })
      return
    }

    if (audioEvent.type === 'speech-candidate') {
      console.debug('[speaker-session] VAD candidate', {
        phase: phaseRef.current,
        revision: turnRevisionRef.current,
        turnId: turnIdRef.current,
      })
      if (phaseRef.current === 'grace' && turnIdRef.current) {
        sendEvent('input.speech_candidate', {
          turnId: turnIdRef.current,
          turnRevision: turnRevisionRef.current,
        })
      }
      return
    }
    if (audioEvent.type === 'speech-candidate-cancelled') {
      console.debug('[speaker-session] VAD candidate cancelled', {
        phase: phaseRef.current,
        revision: turnRevisionRef.current,
        turnId: turnIdRef.current,
      })
      if (phaseRef.current === 'grace' && turnIdRef.current) {
        sendEvent('input.speech_candidate_cancelled', {
          turnId: turnIdRef.current,
          turnRevision: turnRevisionRef.current,
        })
      }
      return
    }
    if (audioEvent.type === 'speech-start') {
      let reopened = false
      if (phaseRef.current === 'grace' && turnIdRef.current) {
        reopened = true
        turnRevisionRef.current += 1
      } else {
        if (phaseRef.current === 'responding') {
          pendingInterruptedGenerationRef.current = generationRef.current
          playerRef.current?.clear()
          generationRef.current = null
          binaryGenerationRef.current = null
        }
        turnIdRef.current =
          globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
        turnRevisionRef.current = 0
      }
      phaseRef.current = 'capturing'
      console.debug('[speaker-session] VAD speech start', {
        reopened,
        revision: turnRevisionRef.current,
        turnId: turnIdRef.current,
      })
      setStatus('hearing')
      sendEvent('input.speech_started', {
        turnId: turnIdRef.current,
        turnRevision: turnRevisionRef.current,
        reopened,
      })
      return
    }
    if (audioEvent.type === 'audio-frame') {
      if (
        phaseRef.current !== 'capturing' ||
        !socket ||
        socket.readyState !== WebSocket.OPEN
      ) {
        console.debug('[speaker-session] dropped microphone frame', {
          phase: phaseRef.current,
          socketState: socket?.readyState,
        })
        return
      }
      if (socket.bufferedAmount > maximumBufferedBytes) {
        setError('The speaker connection cannot keep up with microphone audio.')
        setStatus('error')
        socket.close(1011, 'Input backpressure limit exceeded')
        return
      }
      socket.send(float32ToPcm16(audioEvent.audio))
      return
    }
    if (
      (audioEvent.type === 'speech-end' || audioEvent.type === 'input-limit') &&
      phaseRef.current === 'capturing' &&
      turnIdRef.current
    ) {
      sendEvent(
        audioEvent.type === 'speech-end'
          ? 'input.speech_soft_ended'
          : 'input.limit_reached',
        {
          turnId: turnIdRef.current,
          turnRevision: turnRevisionRef.current,
        }
      )
      phaseRef.current = 'grace'
      console.debug('[speaker-session] VAD soft end', {
        inputLimit: audioEvent.type === 'input-limit',
        revision: turnRevisionRef.current,
        turnId: turnIdRef.current,
      })
      setStatus('thinking')
    }
  }, [])

  return {
    audioLevel,
    canCapture:
      enabled &&
      isReady &&
      !isLanguageUpdating &&
      sessionChatKeyRef.current === activeChatKey,
    canChangeLanguage:
      enabled && isReady && !isLanguageUpdating && status === 'listening',
    error,
    isPlaying,
    isLanguageUpdating,
    latestAssistantTranscript:
      sessionChatKeyRef.current === activeChatKey
        ? latestAssistantTranscript
        : null,
    latestUserTranscript:
      sessionChatKeyRef.current === activeChatKey ? latestUserTranscript : null,
    onAudioEvent,
    status,
  }
}
