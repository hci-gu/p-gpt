import { speakerOutputSampleRate } from '@/src/lib/speaker-audio'

type Segment = {
  chunkCount: number
  done: boolean
  generation: number
  id: string
  minimumSchedulingLeadMilliseconds: number
  pendingSources: number
  text: string
  underrunCount: number
}

export type SpeakerPlaybackDiagnostic = {
  activity: 'playback_buffer_underrun' | 'playback_segment_scheduled'
  chunkCount?: number
  generation: number
  minimumSchedulingLeadMilliseconds?: number
  schedulingLeadMilliseconds?: number
  segmentId: string
  underrunCount?: number
}

type PlaybackCallbacks = {
  onDiagnostic?: (diagnostic: SpeakerPlaybackDiagnostic) => void
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

const schedulingLeadSeconds = 0.08

export class SpeakerPcmPlayback {
  private readonly callbacks: PlaybackCallbacks
  private context: AudioContext | null = null
  private gain: GainNode | null = null
  private nextStartTime = 0
  private schedulingQueue: Promise<void> = Promise.resolve()
  private playbackEpoch = 0
  private timelineGeneration: number | null = null
  private hasScheduledAudio = false
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
    if (this.timelineGeneration !== generation && this.sources.size === 0) {
      this.timelineGeneration = generation
      this.hasScheduledAudio = false
      this.nextStartTime = this.context?.currentTime ?? 0
    }
    const segment: Segment = {
      chunkCount: 0,
      done: false,
      generation,
      id,
      minimumSchedulingLeadMilliseconds: Number.POSITIVE_INFINITY,
      pendingSources: 0,
      text,
      underrunCount: 0,
    }
    this.activeSegment = segment
    this.segments.set(id, segment)
  }

  pushPcm16(generation: number, value: ArrayBuffer): Promise<void> {
    const segment = this.activeSegment
    if (!segment || segment.generation !== generation || value.byteLength < 2) {
      return Promise.resolve()
    }
    const epoch = this.playbackEpoch
    segment.chunkCount += 1
    // Count queued chunks immediately. A segment_done control message may be
    // received before the async AudioContext scheduling work has completed.
    segment.pendingSources += 1
    const schedulingTask = this.schedulingQueue.then(() =>
      this.schedulePcm16(epoch, segment, value)
    )
    // Keep later chunks ordered even if one scheduling operation fails. The
    // returned task still rejects so the session can surface the playback error.
    this.schedulingQueue = schedulingTask.catch(() => undefined)
    return schedulingTask
  }

  private async schedulePcm16(
    epoch: number,
    segment: Segment,
    value: ArrayBuffer
  ) {
    if (epoch !== this.playbackEpoch || !this.segments.has(segment.id)) {
      return
    }
    const context = this.ensureContext()
    await context.resume()
    if (epoch !== this.playbackEpoch || !this.segments.has(segment.id)) {
      return
    }
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
    this.sources.add(source)
    this.callbacks.onPlayingChange(true)
    const previousEndTime = this.nextStartTime
    const wasUnderrun =
      this.hasScheduledAudio && previousEndTime <= context.currentTime
    const startTime = wasUnderrun
      ? context.currentTime + schedulingLeadSeconds
      : Math.max(
          previousEndTime,
          context.currentTime + (this.hasScheduledAudio ? 0 : schedulingLeadSeconds)
        )
    const schedulingLeadMilliseconds =
      (startTime - context.currentTime) * 1_000
    segment.minimumSchedulingLeadMilliseconds = Math.min(
      segment.minimumSchedulingLeadMilliseconds,
      schedulingLeadMilliseconds
    )
    if (wasUnderrun) {
      segment.underrunCount += 1
      this.callbacks.onDiagnostic?.({
        activity: 'playback_buffer_underrun',
        generation: segment.generation,
        schedulingLeadMilliseconds,
        segmentId: segment.id,
      })
    }
    source.start(startTime)
    this.nextStartTime = startTime + audioBuffer.duration
    this.hasScheduledAudio = true
    source.onended = () => {
      this.sources.delete(source)
      segment.pendingSources = Math.max(0, segment.pendingSources - 1)
      this.maybeCompleteSegment(segment)
      const hasQueuedAudio = [...this.segments.values()].some(
        (value) => value.pendingSources > 0
      )
      if (this.sources.size === 0 && !hasQueuedAudio) {
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
    const scheduledThroughSegment = this.schedulingQueue
    void scheduledThroughSegment.then(() => {
      if (!this.segments.has(segment.id)) {
        return
      }
      this.callbacks.onDiagnostic?.({
        activity: 'playback_segment_scheduled',
        chunkCount: segment.chunkCount,
        generation: segment.generation,
        minimumSchedulingLeadMilliseconds: Number.isFinite(
          segment.minimumSchedulingLeadMilliseconds
        )
          ? segment.minimumSchedulingLeadMilliseconds
          : undefined,
        segmentId: segment.id,
        underrunCount: segment.underrunCount,
      })
    })
    this.maybeCompleteSegment(segment)
  }

  endResponse(generation: number) {
    this.responseAudioDone.add(generation)
    this.maybeCompleteResponse(generation)
  }

  clear() {
    this.playbackEpoch += 1
    this.schedulingQueue = Promise.resolve()
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
    this.timelineGeneration = null
    this.hasScheduledAudio = false
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
