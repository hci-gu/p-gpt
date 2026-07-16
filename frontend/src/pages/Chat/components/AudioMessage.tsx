'use client'

import { BotIcon } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

const pcmSampleRate = 24_000
const waveBars = Array.from({ length: 18 }, (_, index) => index)

const createSilentLevels = () => waveBars.map(() => 0.12)

type BrowserWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext
  }

const getAudioContextConstructor = () =>
  window.AudioContext ?? (window as BrowserWindow).webkitAudioContext

const mergeCarry = (carry: Uint8Array | null, chunk: Uint8Array) => {
  if (!carry) {
    return chunk
  }

  const merged = new Uint8Array(carry.byteLength + chunk.byteLength)
  merged.set(carry)
  merged.set(chunk, carry.byteLength)
  return merged
}

const decodePcm16Chunk = (
  audioContext: AudioContext,
  chunk: Uint8Array,
  carry: Uint8Array | null
) => {
  const bytes = mergeCarry(carry, chunk)
  const usableByteLength = bytes.byteLength - (bytes.byteLength % 2)
  const nextCarry =
    usableByteLength === bytes.byteLength ? null : bytes.slice(usableByteLength)
  const sampleCount = usableByteLength / 2
  const audioBuffer = audioContext.createBuffer(
    1,
    sampleCount,
    pcmSampleRate
  )
  const channel = audioBuffer.getChannelData(0)
  const view = new DataView(bytes.buffer, bytes.byteOffset, usableByteLength)
  let totalLevel = 0

  for (let index = 0; index < sampleCount; index += 1) {
    const sample = view.getInt16(index * 2, true) / 32_768
    channel[index] = sample
    totalLevel += Math.abs(sample)
  }

  return {
    audioBuffer,
    carry: nextCarry,
    level: sampleCount ? Math.min(1, totalLevel / sampleCount) : 0,
  }
}

export const AudioMessage = ({
  onEnded,
  onError,
  onLevelChange,
  onPlaybackStart,
  src,
  volume,
}: {
  onEnded: () => void
  onError: () => void
  onLevelChange?: (level: number) => void
  onPlaybackStart?: () => void
  src: string
  volume: number
}) => {
  const activeSourceCountRef = useRef(0)
  const animationRef = useRef<number | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const currentLevelRef = useRef(0)
  const gainNodeRef = useRef<GainNode | null>(null)
  const initialVolumeRef = useRef(volume)
  const hasFinishedRef = useRef(false)
  const hasStartedPlaybackRef = useRef(false)
  const nextStartTimeRef = useRef(0)
  const onEndedRef = useRef(onEnded)
  const onErrorRef = useRef(onError)
  const onLevelChangeRef = useRef(onLevelChange)
  const onPlaybackStartRef = useRef(onPlaybackStart)
  const streamDoneRef = useRef(false)
  const sourcesRef = useRef<AudioBufferSourceNode[]>([])
  const [level, setLevel] = useState(0)
  const [barLevels, setBarLevels] = useState(createSilentLevels)

  onEndedRef.current = onEnded
  onErrorRef.current = onError
  onLevelChangeRef.current = onLevelChange
  onPlaybackStartRef.current = onPlaybackStart

  useEffect(() => {
    const gainNode = gainNodeRef.current

    if (gainNode) {
      gainNode.gain.setValueAtTime(
        Math.min(1, Math.max(0, volume)),
        gainNode.context.currentTime
      )
    }
  }, [volume])

  useEffect(() => {
    const AudioContextConstructor = getAudioContextConstructor()
    const abortController = new AbortController()

    if (!AudioContextConstructor) {
      onErrorRef.current()
      return undefined
    }

    const audioContext = new AudioContextConstructor({
      sampleRate: pcmSampleRate,
    })
    const gainNode = audioContext.createGain()
    gainNode.gain.value = Math.min(1, Math.max(0, initialVolumeRef.current))
    gainNode.connect(audioContext.destination)
    audioContextRef.current = audioContext
    gainNodeRef.current = gainNode
    nextStartTimeRef.current = audioContext.currentTime + 0.08

    const stopVisuals = () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      currentLevelRef.current = 0
      onLevelChangeRef.current?.(0)
      setLevel(0)
      setBarLevels(createSilentLevels())
    }

    const finishPlayback = () => {
      if (hasFinishedRef.current) {
        return
      }

      hasFinishedRef.current = true
      stopVisuals()
      onEndedRef.current()
    }

    const maybeFinishPlayback = () => {
      if (streamDoneRef.current && activeSourceCountRef.current === 0) {
        finishPlayback()
      }
    }

    const startVisuals = () => {
      const tick = () => {
        const time = audioContext.currentTime
        const currentLevel = currentLevelRef.current
        const levels = waveBars.map((_, index) => {
          const slowWave = Math.sin(time * 8 + index * 0.8)
          const fastWave = Math.sin(time * 17 + index * 1.7)
          const movement = 0.36 + slowWave * 0.22 + fastWave * 0.14

          return Math.max(
            0.1,
            Math.min(1, 0.1 + currentLevel * 2.2 + movement * currentLevel)
          )
        })
        const average =
          levels.reduce((sum, value) => sum + value, 0) / levels.length

        onLevelChangeRef.current?.(currentLevel)
        currentLevelRef.current *= 0.92
        setBarLevels(levels)
        setLevel(average)
        animationRef.current = requestAnimationFrame(tick)
      }

      tick()
    }

    const scheduleAudioBuffer = (audioBuffer: AudioBuffer) => {
      if (!audioBuffer.length) {
        return
      }

      const source = audioContext.createBufferSource()
      const startTime = Math.max(
        audioContext.currentTime + 0.03,
        nextStartTimeRef.current
      )

      source.buffer = audioBuffer
      source.connect(gainNode)
      activeSourceCountRef.current += 1
      sourcesRef.current.push(source)
      source.onended = () => {
        activeSourceCountRef.current -= 1
        sourcesRef.current = sourcesRef.current.filter((item) => item !== source)
        maybeFinishPlayback()
      }
      source.start(startTime)
      if (!hasStartedPlaybackRef.current) {
        hasStartedPlaybackRef.current = true
        onPlaybackStartRef.current?.()
      }
      nextStartTimeRef.current = startTime + audioBuffer.duration
    }

    const playStream = async () => {
      let carry: Uint8Array | null = null
      let receivedAnyAudio = false

      try {
        await audioContext.resume()
        startVisuals()

        const response = await fetch(src, {
          headers: { Accept: 'audio/pcm' },
          signal: abortController.signal,
        })

        if (!response.ok || !response.body) {
          throw new Error(`Audio stream failed with status ${response.status}`)
        }

        const reader = response.body.getReader()

        while (true) {
          const { done, value } = await reader.read()

          if (done) {
            break
          }

          if (!value?.byteLength) {
            continue
          }

          receivedAnyAudio = true

          const decoded = decodePcm16Chunk(audioContext, value, carry)
          carry = decoded.carry
          currentLevelRef.current = Math.max(
            currentLevelRef.current,
            Math.min(1, decoded.level * 5)
          )
          scheduleAudioBuffer(decoded.audioBuffer)
        }

        streamDoneRef.current = true
        if (!receivedAnyAudio) {
          throw new Error('Audio stream completed without audio.')
        }
        maybeFinishPlayback()
      } catch {
        if (abortController.signal.aborted) {
          return
        }

        stopVisuals()
        onErrorRef.current()
      }
    }

    void playStream()

    return () => {
      abortController.abort()
      stopVisuals()
      for (const source of sourcesRef.current) {
        try {
          source.stop()
        } catch {
          // Source may already have ended.
        }
      }
      sourcesRef.current = []
      gainNodeRef.current = null
      gainNode.disconnect()
      void audioContext.close()
    }
  }, [src])

  const pulseScale = 1 + level * 0.18

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div
          className="flex size-12 shrink-0 items-center justify-center rounded-full border bg-primary text-primary-foreground shadow-sm transition-transform duration-75"
          style={{ transform: `scale(${pulseScale})` }}
        >
          <BotIcon className="size-5" />
        </div>
        <div className="flex h-12 flex-1 items-center gap-1">
          {waveBars.map((bar) => {
            const height = 8 + barLevels[bar] * 42

            return (
              <div
                className="w-1 rounded-full bg-primary/80 transition-[height] duration-75"
                key={bar}
                style={{ height }}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}
