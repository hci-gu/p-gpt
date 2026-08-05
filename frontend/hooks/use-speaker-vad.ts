import { useCallback, useEffect, useRef, useState } from 'react'
import {
  discardSpeakerAudioEvent,
  speakerAudioSampleRate,
  speakerVadFrameSamples,
  type SpeakerAudioConsumer,
  type SpeakerAudioEvent,
} from '@/src/lib/speaker-audio'

type BrowserWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext
  }

type SpeakerVadStatus = 'idle' | 'loading' | 'monitoring' | 'error'

type VadWorkerResponse =
  | { type: 'ready' }
  | { active: boolean; type: 'activity-change' }
  | { type: 'speech-start' }
  | { type: 'speech-candidate' }
  | { type: 'speech-candidate-cancelled' }
  | { audio: Float32Array; type: 'audio-frame' }
  | { probability: number; type: 'speech-probability' }
  | { type: 'speech-end' }
  | { type: 'input-limit' }
  | { error: string; type: 'error' }

type UseSpeakerVadOptions = {
  enabled: boolean
  paused?: boolean
  onAudioEvent?: SpeakerAudioConsumer
}

const getAudioContextConstructor = () =>
  window.AudioContext ?? (window as BrowserWindow).webkitAudioContext

const isSupported = () =>
  typeof window !== 'undefined' &&
  typeof Worker !== 'undefined' &&
  Boolean(getAudioContextConstructor()) &&
  Boolean(navigator.mediaDevices?.getUserMedia)

const resampleToTargetRate = (audio: Float32Array, sourceSampleRate: number) => {
  if (sourceSampleRate === speakerAudioSampleRate) {
    return audio.slice()
  }

  const sampleRateRatio = sourceSampleRate / speakerAudioSampleRate
  const outputLength = Math.max(1, Math.round(audio.length / sampleRateRatio))
  const output = new Float32Array(outputLength)

  for (let index = 0; index < outputLength; index += 1) {
    const sourceIndex = index * sampleRateRatio
    const beforeIndex = Math.floor(sourceIndex)
    const afterIndex = Math.min(audio.length - 1, beforeIndex + 1)
    const weight = sourceIndex - beforeIndex
    const before = audio[beforeIndex] ?? 0
    const after = audio[afterIndex] ?? before

    output[index] = before + (after - before) * weight
  }

  return output
}

const appendAudio = (current: Float32Array, incoming: Float32Array) => {
  const output = new Float32Array(current.length + incoming.length)
  output.set(current)
  output.set(incoming, current.length)
  return output
}

export const useSpeakerVad = ({
  enabled,
  paused = false,
  onAudioEvent = discardSpeakerAudioEvent,
}: UseSpeakerVadOptions) => {
  const [error, setError] = useState<string | null>(null)
  const [isMuted, setIsMuted] = useState(true)
  const [isVoiceActive, setIsVoiceActive] = useState(false)
  const [status, setStatus] = useState<SpeakerVadStatus>('idle')
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioBufferRef = useRef(new Float32Array())
  const isRunningRef = useRef(false)
  const onAudioEventRef = useRef(onAudioEvent)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const wasEnabledRef = useRef(false)

  onAudioEventRef.current = onAudioEvent

  const cleanupAudio = useCallback(async () => {
    isRunningRef.current = false
    audioBufferRef.current = new Float32Array()
    setIsVoiceActive(false)

    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current.onaudioprocess = null
      processorRef.current = null
    }

    sourceRef.current?.disconnect()
    sourceRef.current = null

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop()
      }
      streamRef.current = null
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close()
      audioContextRef.current = null
    }

    workerRef.current?.postMessage({ type: 'reset' })
  }, [])

  const disposeWorker = useCallback(() => {
    workerRef.current?.terminate()
    workerRef.current = null
  }, [])

  const getWorker = useCallback(() => {
    if (workerRef.current) {
      return workerRef.current
    }

    const worker = new Worker(
      new URL('../src/workers/vad-worker.ts', import.meta.url),
      { type: 'module' }
    )

    worker.addEventListener('message', (event: MessageEvent<VadWorkerResponse>) => {
      const message = event.data

      if (message.type === 'ready') {
        setStatus('monitoring')
        return
      }

      if (message.type === 'error') {
        setError(message.error)
        setStatus('error')
        setIsMuted(true)
        void cleanupAudio()
        return
      }

      if (message.type === 'activity-change') {
        setIsVoiceActive(message.active)
        return
      }

      if (message.type === 'speech-probability') {
        return
      }

      const audioEvent: SpeakerAudioEvent =
        message.type === 'audio-frame'
          ? {
              audio: message.audio,
              sampleRate: speakerAudioSampleRate,
              type: 'audio-frame',
            }
          : { type: message.type }
      onAudioEventRef.current(audioEvent)
    })

    workerRef.current = worker
    return worker
  }, [cleanupAudio])

  const startMonitoring = useCallback(async () => {
    if (isRunningRef.current || !enabled || paused) {
      return
    }

    if (!isSupported()) {
      setError('Voice activity detection is not supported in this browser.')
      setStatus('error')
      setIsMuted(true)
      return
    }

    const AudioContextConstructor = getAudioContextConstructor()
    if (!AudioContextConstructor) {
      return
    }

    setError(null)
    setStatus('loading')

    try {
      const worker = getWorker()
      worker.postMessage({ type: 'init' })

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      const audioContext = new AudioContextConstructor()
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)

      streamRef.current = stream
      audioContextRef.current = audioContext
      sourceRef.current = source
      processorRef.current = processor
      isRunningRef.current = true

      processor.onaudioprocess = (event) => {
        if (!isRunningRef.current) {
          return
        }

        const sourceAudio = new Float32Array(event.inputBuffer.getChannelData(0))
        const audio = resampleToTargetRate(sourceAudio, audioContext.sampleRate)
        audioBufferRef.current = appendAudio(audioBufferRef.current, audio)

        while (audioBufferRef.current.length >= speakerVadFrameSamples) {
          const frame = audioBufferRef.current.slice(0, speakerVadFrameSamples)
          audioBufferRef.current = audioBufferRef.current.slice(speakerVadFrameSamples)
          worker.postMessage({ audio: frame, type: 'process' }, [frame.buffer])
        }
      }

      source.connect(processor)
      processor.connect(audioContext.destination)
      await audioContext.resume()
    } catch (caughtError) {
      await cleanupAudio()
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'Could not start microphone monitoring.'
      )
      setStatus('error')
      setIsMuted(true)
    }
  }, [cleanupAudio, enabled, getWorker, paused])

  const mute = useCallback(() => {
    setIsMuted(true)
    setStatus('idle')
    void cleanupAudio()
  }, [cleanupAudio])

  const unmute = useCallback(() => {
    setIsMuted(false)
  }, [])

  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      setIsMuted(false)
    }
    wasEnabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    if (!enabled || isMuted || paused) {
      void cleanupAudio()
      if (!enabled) {
        disposeWorker()
        setStatus('idle')
      } else if (paused) {
        setStatus('idle')
      }
      return
    }

    void startMonitoring()
  }, [cleanupAudio, disposeWorker, enabled, isMuted, paused, startMonitoring])

  useEffect(
    () => () => {
      void cleanupAudio()
      disposeWorker()
    },
    [cleanupAudio, disposeWorker]
  )

  return {
    error,
    isMuted: isMuted || paused || !enabled,
    isVoiceActive,
    mute,
    status,
    unmute,
  }
}
