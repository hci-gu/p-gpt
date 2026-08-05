import { speakerOutputSampleRate } from '@/src/lib/speaker-audio'

type Segment = {
  done: boolean
  generation: number
  id: string
  pendingSources: number
  text: string
}

type PlaybackCallbacks = {
  onLevelChange: (level: number) => void
  onPlayingChange: (playing: boolean) => void
  onResponseCompleted: (generation: number) => void
  onSegmentCompleted: (generation: number, segmentId: string, text: string) => void
}

type BrowserWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext
  }

const getAudioContextConstructor = () =>
  window.AudioContext ?? (window as BrowserWindow).webkitAudioContext

export class SpeakerPcmPlayback {
  private readonly callbacks: PlaybackCallbacks
  private context: AudioContext | null = null
  private gain: GainNode | null = null
  private nextStartTime = 0
  private activeSegment: Segment | null = null
  private segments = new Map<string, Segment>()
  private sources = new Set<AudioBufferSourceNode>()
  private responseAudioDone = new Set<number>()
  private volume: number

  constructor(volume: number, callbacks: PlaybackCallbacks) {
    this.volume = volume
    this.callbacks = callbacks
  }

  setVolume(volume: number) {
    this.volume = Math.min(1, Math.max(0, volume))
    if (this.gain && this.context) {
      this.gain.gain.setValueAtTime(this.volume, this.context.currentTime)
    }
  }

  beginSegment(generation: number, id: string, text: string) {
    const segment: Segment = {
      done: false,
      generation,
      id,
      pendingSources: 0,
      text,
    }
    this.activeSegment = segment
    this.segments.set(id, segment)
  }

  async pushPcm16(generation: number, value: ArrayBuffer) {
    const segment = this.activeSegment
    if (!segment || segment.generation !== generation || value.byteLength < 2) {
      return
    }
    const context = this.ensureContext()
    await context.resume()
    const usableBytes = value.byteLength - (value.byteLength % 2)
    const sampleCount = usableBytes / 2
    const audioBuffer = context.createBuffer(1, sampleCount, speakerOutputSampleRate)
    const output = audioBuffer.getChannelData(0)
    const input = new DataView(value, 0, usableBytes)
    let totalLevel = 0
    for (let index = 0; index < sampleCount; index += 1) {
      const sample = input.getInt16(index * 2, true) / 32_768
      output[index] = sample
      totalLevel += Math.abs(sample)
    }
    this.callbacks.onLevelChange(
      sampleCount ? Math.min(1, totalLevel / sampleCount) : 0
    )

    const source = context.createBufferSource()
    source.buffer = audioBuffer
    source.connect(this.gain as GainNode)
    segment.pendingSources += 1
    this.sources.add(source)
    this.callbacks.onPlayingChange(true)
    const startTime = Math.max(
      this.nextStartTime,
      context.currentTime + (this.sources.size === 1 ? 0.08 : 0)
    )
    source.start(startTime)
    this.nextStartTime = startTime + audioBuffer.duration
    source.onended = () => {
      this.sources.delete(source)
      segment.pendingSources = Math.max(0, segment.pendingSources - 1)
      this.maybeCompleteSegment(segment)
      if (this.sources.size === 0) {
        this.callbacks.onLevelChange(0)
        this.callbacks.onPlayingChange(false)
      }
    }
  }

  endSegment(generation: number, id: string) {
    const segment = this.segments.get(id)
    if (!segment || segment.generation !== generation) {
      return
    }
    segment.done = true
    if (this.activeSegment?.id === id) {
      this.activeSegment = null
    }
    this.maybeCompleteSegment(segment)
  }

  endResponse(generation: number) {
    this.responseAudioDone.add(generation)
    this.maybeCompleteResponse(generation)
  }

  clear() {
    for (const source of this.sources) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // A source may have ended between the stale-generation check and stop.
      }
    }
    this.sources.clear()
    this.segments.clear()
    this.responseAudioDone.clear()
    this.activeSegment = null
    this.nextStartTime = this.context?.currentTime ?? 0
    this.callbacks.onLevelChange(0)
    this.callbacks.onPlayingChange(false)
  }

  async dispose() {
    this.clear()
    if (this.context) {
      await this.context.close()
      this.context = null
      this.gain = null
    }
  }

  private ensureContext() {
    if (this.context) {
      return this.context
    }
    const AudioContextConstructor = getAudioContextConstructor()
    if (!AudioContextConstructor) {
      throw new Error('Web Audio is not supported in this browser.')
    }
    this.context = new AudioContextConstructor({ sampleRate: speakerOutputSampleRate })
    this.gain = this.context.createGain()
    this.gain.gain.value = this.volume
    this.gain.connect(this.context.destination)
    this.nextStartTime = this.context.currentTime
    return this.context
  }

  private maybeCompleteSegment(segment: Segment) {
    if (!segment.done || segment.pendingSources > 0 || !this.segments.has(segment.id)) {
      return
    }
    this.segments.delete(segment.id)
    this.callbacks.onSegmentCompleted(
      segment.generation,
      segment.id,
      segment.text
    )
    this.maybeCompleteResponse(segment.generation)
  }

  private maybeCompleteResponse(generation: number) {
    if (!this.responseAudioDone.has(generation)) {
      return
    }
    const hasPendingSegment = [...this.segments.values()].some(
      (segment) => segment.generation === generation
    )
    if (hasPendingSegment) {
      return
    }
    this.responseAudioDone.delete(generation)
    this.callbacks.onResponseCompleted(generation)
  }
}
