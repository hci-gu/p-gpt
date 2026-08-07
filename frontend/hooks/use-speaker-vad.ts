import { useCallback, useEffect, useRef, useState } from 'react'
import {
  SpeakerRecoveryWindow,
  speakerCaptureStallMilliseconds,
  speakerMaximumPendingVadFrames,
} from '@/src/lib/speaker-capture-recovery'
import {
  discardSpeakerAudioEvent,
  speakerAudioSampleRate,
  speakerVadFrameSamples,
  type SpeakerAudioConsumer,
  type SpeakerAudioEvent,
} from '@/src/lib/speaker-audio'
import {
  describeSpeakerVadConfig,
  speakerVadConfig,
  type SpeakerVadDetectionProfile,
} from '@/src/lib/speaker-vad-config'

type BrowserWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext
  }

type SpeakerVadStatus = 'idle' | 'loading' | 'monitoring' | 'error'

type VadWorkerResponse =
  | { epoch: number; type: 'ready' | 'reset-complete' }
  | { active: boolean; epoch: number; sequence: number; type: 'activity-change' }
  | {
      diagnosticDetail?: string
      epoch: number
      sequence: number
      type:
        | 'speech-start'
        | 'speech-candidate'
        | 'speech-candidate-cancelled'
        | 'speech-end'
        | 'input-limit'
    }
  | { audio: Float32Array; epoch: number; sequence: number; type: 'audio-frame' }
  | {
      epoch: number
      probability: number
      processingMilliseconds: number
      queueDelayMilliseconds: number
      sequence: number
      type: 'speech-probability'
    }
  | { epoch: number; error: string; type: 'error' }

type UseSpeakerVadOptions = {
  detectionProfile?: SpeakerVadDetectionProfile
  enabled: boolean
  paused?: boolean
  onAudioEvent?: SpeakerAudioConsumer
}

type WorkerTimingWindow = {
  count: number
  processingMaximum: number
  processingTotal: number
  queueMaximum: number
  queueTotal: number
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

const emptyWorkerTimingWindow = (): WorkerTimingWindow => ({
  count: 0,
  processingMaximum: 0,
  processingTotal: 0,
  queueMaximum: 0,
  queueTotal: 0,
})

export const useSpeakerVad = ({
  detectionProfile = 'start',
  enabled,
  paused = false,
  onAudioEvent = discardSpeakerAudioEvent,
}: UseSpeakerVadOptions) => {
  const [error, setError] = useState<string | null>(null)
  const [isMuted, setIsMuted] = useState(true)
  const [isVoiceActive, setIsVoiceActive] = useState(false)
  const [status, setStatus] = useState<SpeakerVadStatus>('idle')
  const [restartToken, setRestartToken] = useState(0)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioBufferRef = useRef(new Float32Array())
  const candidateActiveRef = useRef(false)
  const captureEpochRef = useRef(0)
  const captureStartedRef = useRef(false)
  const cleanupPromiseRef = useRef<Promise<void>>(Promise.resolve())
  const detectionProfileRef = useRef(detectionProfile)
  const enabledRef = useRef(enabled)
  const isMutedRef = useRef(true)
  const isRunningRef = useRef(false)
  const isVoiceActiveRef = useRef(false)
  const lastCaptureCallbackAtRef = useRef(0)
  const lastWorkerResultAtRef = useRef(0)
  const nextSequenceRef = useRef(0)
  const onAudioEventRef = useRef(onAudioEvent)
  const pausedRef = useRef(paused)
  const pendingFramesRef = useRef(new Map<number, number>())
  const pendingRecoveryRef = useRef<{ count: number; reason: string } | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const recoveryWindowRef = useRef(new SpeakerRecoveryWindow())
  const requestRecoveryRef = useRef<(reason: string, detail: string) => void>(
    () => undefined
  )
  const restartPendingRef = useRef(false)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const staleEventReportAtRef = useRef(0)
  const startPromiseRef = useRef<Promise<void> | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const wasEnabledRef = useRef(false)
  const workerReadyWaiterRef = useRef<{
    epoch: number
    resolve: (ready: boolean) => void
  } | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const workerTimingWindowRef = useRef(emptyWorkerTimingWindow())
  const probabilityWindowRef = useRef({
    count: 0,
    maximum: 0,
    minimum: 1,
    startedAt: 0,
    total: 0,
  })

  detectionProfileRef.current = detectionProfile
  enabledRef.current = enabled
  onAudioEventRef.current = onAudioEvent
  pausedRef.current = paused

  const emitDiagnostic = useCallback(
    (
      diagnostic: Omit<
        Extract<SpeakerAudioEvent, { type: 'vad-diagnostic' }>,
        'type'
      >
    ) => {
      const event = { ...diagnostic, type: 'vad-diagnostic' } as const
      console.debug('[speaker-vad]', event)
      onAudioEventRef.current(event)
    },
    []
  )

  const resetProbabilityWindow = useCallback(() => {
    probabilityWindowRef.current = {
      count: 0,
      maximum: 0,
      minimum: 1,
      startedAt: performance.now(),
      total: 0,
    }
    workerTimingWindowRef.current = emptyWorkerTimingWindow()
  }, [])

  const reportStaleEvent = useCallback(
    (eventEpoch: number, eventType: string) => {
      const now = performance.now()
      if (now - staleEventReportAtRef.current < 1_000) {
        return
      }
      staleEventReportAtRef.current = now
      emitDiagnostic({
        activity: 'stale_event',
        captureEpoch: captureEpochRef.current,
        detail: `event=${eventType};event_epoch=${eventEpoch}`,
        detectionProfile: detectionProfileRef.current,
      })
    },
    [emitDiagnostic]
  )

  const disposeWorker = useCallback(() => {
    workerReadyWaiterRef.current?.resolve(false)
    workerReadyWaiterRef.current = null
    workerRef.current?.terminate()
    workerRef.current = null
    pendingFramesRef.current.clear()
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
      if (message.epoch !== captureEpochRef.current) {
        reportStaleEvent(message.epoch, message.type)
        return
      }

      if (message.type === 'ready') {
        if (workerReadyWaiterRef.current?.epoch === message.epoch) {
          workerReadyWaiterRef.current.resolve(true)
          workerReadyWaiterRef.current = null
        }
        emitDiagnostic({
          activity: 'worker_ready',
          captureEpoch: message.epoch,
          detectionProfile: detectionProfileRef.current,
        })
        return
      }
      if (message.type === 'reset-complete') {
        return
      }
      if (message.type === 'error') {
        emitDiagnostic({
          activity: message.error.includes('timed out')
            ? 'inference_timeout'
            : 'worker_error',
          captureEpoch: message.epoch,
          detail: message.error.slice(0, 256),
          detectionProfile: detectionProfileRef.current,
          pendingFrameCount: pendingFramesRef.current.size,
        })
        disposeWorker()
        requestRecoveryRef.current('worker_error', message.error.slice(0, 256))
        return
      }

      if (message.type === 'speech-probability') {
        pendingFramesRef.current.delete(message.sequence)
        lastWorkerResultAtRef.current = performance.now()
        const timing = workerTimingWindowRef.current
        timing.count += 1
        timing.processingTotal += message.processingMilliseconds
        timing.processingMaximum = Math.max(
          timing.processingMaximum,
          message.processingMilliseconds
        )
        timing.queueTotal += message.queueDelayMilliseconds
        timing.queueMaximum = Math.max(
          timing.queueMaximum,
          message.queueDelayMilliseconds
        )

        if (!Number.isFinite(message.probability)) {
          return
        }
        const probability = Math.min(1, Math.max(0, message.probability))
        const window = probabilityWindowRef.current
        if (window.count === 0) {
          window.startedAt = performance.now()
        }
        window.count += 1
        window.total += probability
        window.minimum = Math.min(window.minimum, probability)
        window.maximum = Math.max(window.maximum, probability)
        if (performance.now() - window.startedAt >= 1_000) {
          emitDiagnostic({
            activity: 'probability_summary',
            captureEpoch: message.epoch,
            detectionProfile: detectionProfileRef.current,
            pendingFrameCount: pendingFramesRef.current.size,
            probabilityAverage: window.total / window.count,
            probabilityMax: window.maximum,
            probabilityMin: window.minimum,
            processingAverageMilliseconds:
              timing.count > 0 ? timing.processingTotal / timing.count : 0,
            processingMaximumMilliseconds: timing.processingMaximum,
            queueDelayAverageMilliseconds:
              timing.count > 0 ? timing.queueTotal / timing.count : 0,
            queueDelayMaximumMilliseconds: timing.queueMaximum,
            sampleCount: window.count,
          })
          resetProbabilityWindow()
        }
        return
      }

      if (message.type === 'activity-change') {
        console.debug('[speaker-vad] activity-change', message.active)
        isVoiceActiveRef.current = message.active
        setIsVoiceActive(message.active)
        return
      }

      if (
        message.type === 'speech-candidate' ||
        message.type === 'speech-candidate-cancelled' ||
        message.type === 'speech-start' ||
        message.type === 'speech-end'
      ) {
        if (message.type === 'speech-candidate') {
          candidateActiveRef.current = true
        } else if (
          message.type === 'speech-candidate-cancelled' ||
          message.type === 'speech-start'
        ) {
          candidateActiveRef.current = false
        }
        if (message.type === 'speech-end') {
          isVoiceActiveRef.current = false
        }
        emitDiagnostic({
          activity: 'vad_state',
          captureEpoch: message.epoch,
          detail: `event=${message.type};sequence=${message.sequence};${message.diagnosticDetail ?? ''}`.slice(
            0,
            256
          ),
          detectionProfile: detectionProfileRef.current,
          pendingFrameCount: pendingFramesRef.current.size,
        })
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
  }, [disposeWorker, emitDiagnostic, reportStaleEvent, resetProbabilityWindow])

  const ensureWorkerReady = useCallback(
    (epoch: number) => {
      const worker = getWorker()
      workerReadyWaiterRef.current?.resolve(false)
      return new Promise<boolean>((resolve) => {
        workerReadyWaiterRef.current = { epoch, resolve }
        worker.postMessage({ epoch, type: 'init' })
      })
    },
    [getWorker]
  )

  const cleanupAudio = useCallback(
    (reason = 'cleanup') => {
      const invalidationEpoch = captureEpochRef.current + 1
      captureEpochRef.current = invalidationEpoch
      const wasRunning = isRunningRef.current || captureStartedRef.current
      const wasCandidateActive = candidateActiveRef.current
      const wasVoiceActive = isVoiceActiveRef.current
      isRunningRef.current = false
      captureStartedRef.current = false
      candidateActiveRef.current = false
      isVoiceActiveRef.current = false
      lastCaptureCallbackAtRef.current = 0
      lastWorkerResultAtRef.current = 0
      audioBufferRef.current = new Float32Array()
      pendingFramesRef.current.clear()
      setIsVoiceActive(false)

      if (wasCandidateActive) {
        onAudioEventRef.current({ type: 'speech-candidate-cancelled' })
      }
      if (wasVoiceActive) {
        onAudioEventRef.current({ type: 'speech-end' })
      }

      workerReadyWaiterRef.current?.resolve(false)
      workerReadyWaiterRef.current = null
      workerRef.current?.postMessage({ epoch: invalidationEpoch, type: 'reset' })

      const processor = processorRef.current
      const source = sourceRef.current
      const stream = streamRef.current
      const audioContext = audioContextRef.current
      processorRef.current = null
      sourceRef.current = null
      streamRef.current = null
      audioContextRef.current = null

      if (wasRunning) {
        emitDiagnostic({
          activity: 'capture_stopped',
          captureEpoch: invalidationEpoch,
          detail: `reason=${reason}`,
          detectionProfile: detectionProfileRef.current,
        })
      }
      resetProbabilityWindow()

      const closeResources = async () => {
        if (processor) {
          processor.onaudioprocess = null
          processor.disconnect()
        }
        source?.disconnect()
        if (stream) {
          for (const track of stream.getTracks()) {
            track.onended = null
            track.stop()
          }
        }
        if (audioContext) {
          audioContext.onstatechange = null
          if (audioContext.state !== 'closed') {
            await audioContext.close()
          }
        }
      }

      const cleanupPromise = cleanupPromiseRef.current.then(
        closeResources,
        closeResources
      )
      cleanupPromiseRef.current = cleanupPromise.catch(() => undefined)
      return cleanupPromise
    },
    [emitDiagnostic, resetProbabilityWindow]
  )

  const requestRecovery = useCallback(
    (reason: string, detail: string) => {
      if (restartPendingRef.current || isMutedRef.current) {
        return
      }
      const recovery = recoveryWindowRef.current.record(performance.now())
      restartPendingRef.current = true
      emitDiagnostic({
        activity: 'capture_recovery_started',
        captureEpoch: captureEpochRef.current,
        detail: `reason=${reason};${detail}`.slice(0, 256),
        detectionProfile: detectionProfileRef.current,
        pendingFrameCount: pendingFramesRef.current.size,
        recoveryCount: recovery.count,
      })

      if (recovery.exhausted) {
        emitDiagnostic({
          activity: 'capture_recovery_exhausted',
          captureEpoch: captureEpochRef.current,
          detail: `reason=${reason}`,
          detectionProfile: detectionProfileRef.current,
          recoveryCount: recovery.count,
        })
        setError(
          'Microphone processing stalled repeatedly. Unmute the microphone to try again.'
        )
        setStatus('error')
        isMutedRef.current = true
        setIsMuted(true)
        void cleanupAudio('recovery_exhausted').finally(() => {
          restartPendingRef.current = false
        })
        return
      }

      pendingRecoveryRef.current = { count: recovery.count, reason }
      const activeStart = startPromiseRef.current
      void cleanupAudio(`recovery_${reason}`).finally(async () => {
        if (activeStart) {
          await activeStart.catch(() => undefined)
        }
        restartPendingRef.current = false
        setRestartToken((current) => current + 1)
      })
    },
    [cleanupAudio, emitDiagnostic]
  )
  requestRecoveryRef.current = requestRecovery

  const startMonitoring = useCallback(() => {
    if (
      startPromiseRef.current ||
      isRunningRef.current ||
      !enabledRef.current ||
      pausedRef.current ||
      isMutedRef.current
    ) {
      return
    }

    if (!isSupported()) {
      setError('Voice activity detection is not supported in this browser.')
      setStatus('error')
      isMutedRef.current = true
      setIsMuted(true)
      return
    }

    const AudioContextConstructor = getAudioContextConstructor()
    if (!AudioContextConstructor) {
      return
    }

    setError(null)
    setStatus('loading')

    const start = async () => {
      await cleanupPromiseRef.current
      if (!enabledRef.current || pausedRef.current || isMutedRef.current) {
        return
      }

      const epoch = captureEpochRef.current + 1
      captureEpochRef.current = epoch
      emitDiagnostic({
        activity: 'vad_config',
        captureEpoch: epoch,
        detail: describeSpeakerVadConfig(speakerVadConfig),
        detectionProfile: detectionProfileRef.current,
      })

      const workerReady = await ensureWorkerReady(epoch)
      if (!workerReady || epoch !== captureEpochRef.current) {
        return
      }

      let stream: MediaStream | null = null
      let audioContext: AudioContext | null = null
      let source: MediaStreamAudioSourceNode | null = null
      let processor: ScriptProcessorNode | null = null
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        })
        if (epoch !== captureEpochRef.current) {
          for (const track of stream.getTracks()) {
            track.stop()
          }
          return
        }

        audioContext = new AudioContextConstructor()
        source = audioContext.createMediaStreamSource(stream)
        processor = audioContext.createScriptProcessor(4096, 1, 1)

        audioContext.onstatechange = () => {
          if (epoch !== captureEpochRef.current) {
            return
          }
          emitDiagnostic({
            activity: 'audio_context',
            captureEpoch: epoch,
            detail: audioContext?.state,
            detectionProfile: detectionProfileRef.current,
          })
        }
        for (const track of stream.getAudioTracks()) {
          track.onended = () => {
            if (epoch !== captureEpochRef.current || !isRunningRef.current) {
              return
            }
            emitDiagnostic({
              activity: 'microphone_ended',
              captureEpoch: epoch,
              detectionProfile: detectionProfileRef.current,
            })
            requestRecoveryRef.current('microphone_ended', 'track_ended=true')
          }
        }

        const captureWorker = workerRef.current
        if (!captureWorker) {
          throw new Error('The voice activity worker stopped before capture began.')
        }
        processor.onaudioprocess = (event) => {
          if (epoch !== captureEpochRef.current || !isRunningRef.current) {
            return
          }
          const now = performance.now()
          lastCaptureCallbackAtRef.current = now

          const sourceAudio = new Float32Array(event.inputBuffer.getChannelData(0))
          const audio = resampleToTargetRate(sourceAudio, audioContext?.sampleRate ?? 0)
          audioBufferRef.current = appendAudio(audioBufferRef.current, audio)

          while (audioBufferRef.current.length >= speakerVadFrameSamples) {
            if (pendingFramesRef.current.size >= speakerMaximumPendingVadFrames) {
              requestRecoveryRef.current(
                'worker_backlog',
                `pending_frames=${pendingFramesRef.current.size}`
              )
              return
            }
            const frame = audioBufferRef.current.slice(0, speakerVadFrameSamples)
            audioBufferRef.current = audioBufferRef.current.slice(
              speakerVadFrameSamples
            )
            const sequence = nextSequenceRef.current + 1
            nextSequenceRef.current = sequence
            pendingFramesRef.current.set(sequence, now)
            captureWorker.postMessage(
              {
                audio: frame,
                capturedAt: now,
                epoch,
                profile: detectionProfileRef.current,
                sequence,
                type: 'process',
              },
              [frame.buffer]
            )
          }
        }

        source.connect(processor)
        processor.connect(audioContext.destination)
        await audioContext.resume()
        if (epoch !== captureEpochRef.current) {
          processor.onaudioprocess = null
          processor.disconnect()
          source.disconnect()
          for (const track of stream.getTracks()) {
            track.stop()
          }
          if (audioContext.state !== 'closed') {
            await audioContext.close()
          }
          return
        }

        streamRef.current = stream
        audioContextRef.current = audioContext
        sourceRef.current = source
        processorRef.current = processor
        const startedAt = performance.now()
        lastCaptureCallbackAtRef.current = startedAt
        lastWorkerResultAtRef.current = startedAt
        captureStartedRef.current = true
        isRunningRef.current = true
        resetProbabilityWindow()
        setStatus('monitoring')
        emitDiagnostic({
          activity: 'capture_started',
          captureEpoch: epoch,
          detail: `source_rate=${audioContext.sampleRate};target_rate=${speakerAudioSampleRate}`,
          detectionProfile: detectionProfileRef.current,
        })
        if (pendingRecoveryRef.current) {
          emitDiagnostic({
            activity: 'capture_recovery_completed',
            captureEpoch: epoch,
            detail: `reason=${pendingRecoveryRef.current.reason}`,
            detectionProfile: detectionProfileRef.current,
            recoveryCount: pendingRecoveryRef.current.count,
          })
          pendingRecoveryRef.current = null
        }
      } catch (caughtError) {
        if (processor) {
          processor.onaudioprocess = null
          processor.disconnect()
        }
        source?.disconnect()
        if (stream) {
          for (const track of stream.getTracks()) {
            track.onended = null
            track.stop()
          }
        }
        if (audioContext && audioContext.state !== 'closed') {
          await audioContext.close().catch(() => undefined)
        }
        if (epoch !== captureEpochRef.current) {
          return
        }
        const message =
          caughtError instanceof Error
            ? caughtError.message
            : 'Could not start microphone monitoring.'
        setError(message)
        setStatus('error')
        isMutedRef.current = true
        setIsMuted(true)
        await cleanupAudio('start_failed')
      }
    }

    const startPromise = start()
    startPromiseRef.current = startPromise
    void startPromise.finally(() => {
      if (startPromiseRef.current === startPromise) {
        startPromiseRef.current = null
      }
    })
  }, [cleanupAudio, emitDiagnostic, ensureWorkerReady, resetProbabilityWindow])

  const mute = useCallback(() => {
    isMutedRef.current = true
    setIsMuted(true)
    setStatus('idle')
    void cleanupAudio('muted')
  }, [cleanupAudio])

  const unmute = useCallback(() => {
    recoveryWindowRef.current.reset()
    pendingRecoveryRef.current = null
    restartPendingRef.current = false
    setError(null)
    isMutedRef.current = false
    setIsMuted(false)
  }, [])

  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      recoveryWindowRef.current.reset()
      isMutedRef.current = false
      setIsMuted(false)
    }
    wasEnabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    if (!enabled || isMuted || paused) {
      void cleanupAudio(!enabled ? 'disabled' : paused ? 'paused' : 'muted')
      if (!enabled) {
        disposeWorker()
        setStatus('idle')
      } else if (paused) {
        setStatus('idle')
      }
      return
    }

    startMonitoring()
  }, [
    cleanupAudio,
    disposeWorker,
    enabled,
    isMuted,
    paused,
    restartToken,
    startMonitoring,
  ])

  useEffect(() => {
    if (!enabled || isMuted || paused) {
      return
    }
    const watchdog = window.setInterval(() => {
      if (
        !isRunningRef.current ||
        !captureStartedRef.current ||
        restartPendingRef.current
      ) {
        return
      }
      const now = performance.now()
      const callbackAge = now - lastCaptureCallbackAtRef.current
      if (callbackAge >= speakerCaptureStallMilliseconds) {
        emitDiagnostic({
          activity: 'capture_stalled',
          captureEpoch: captureEpochRef.current,
          detail: `no_audio_callback_ms=${Math.round(callbackAge)}`,
          detectionProfile: detectionProfileRef.current,
        })
        requestRecovery('capture_callback', `callback_age_ms=${Math.round(callbackAge)}`)
        return
      }

      if (pendingFramesRef.current.size === 0) {
        return
      }
      const oldestPendingAt = Math.min(...pendingFramesRef.current.values())
      const oldestPendingAge = now - oldestPendingAt
      const workerResultAge = now - lastWorkerResultAtRef.current
      if (
        pendingFramesRef.current.size >= speakerMaximumPendingVadFrames ||
        oldestPendingAge >= speakerCaptureStallMilliseconds ||
        workerResultAge >= speakerCaptureStallMilliseconds
      ) {
        emitDiagnostic({
          activity: 'worker_summary',
          captureEpoch: captureEpochRef.current,
          detail: `stalled=true;oldest_pending_ms=${Math.round(oldestPendingAge)};result_age_ms=${Math.round(workerResultAge)}`,
          detectionProfile: detectionProfileRef.current,
          pendingFrameCount: pendingFramesRef.current.size,
        })
        disposeWorker()
        requestRecovery(
          'worker_stalled',
          `pending_frames=${pendingFramesRef.current.size};oldest_pending_ms=${Math.round(oldestPendingAge)}`
        )
      }
    }, 1_000)
    return () => window.clearInterval(watchdog)
  }, [disposeWorker, emitDiagnostic, enabled, isMuted, paused, requestRecovery])

  useEffect(
    () => () => {
      isMutedRef.current = true
      void cleanupAudio('unmounted')
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
