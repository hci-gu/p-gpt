import { afterEach, describe, expect, it, vi } from 'vitest'
import { SpeakerPcmPlayback } from '@/src/lib/speaker-playback'

class FakeSource {
  buffer: { duration: number } | null = null
  onended: (() => void) | null = null
  startTime: number | null = null

  connect() {}
  start(time = 0) {
    this.startTime = time
  }
  stop() {}
  finish() {
    this.onended?.()
  }
}

class FakeAudioContext {
  static sources: FakeSource[] = []
  static instance: FakeAudioContext | null = null
  currentTime = 0
  destination = {}

  constructor() {
    FakeAudioContext.instance = this
  }

  createGain() {
    return {
      connect: () => undefined,
      gain: {
        setValueAtTime: () => undefined,
        value: 1,
      },
    }
  }

  createBuffer(_channels: number, length: number, sampleRate: number) {
    const channel = new Float32Array(length)
    return {
      duration: length / sampleRate,
      getChannelData: () => channel,
    }
  }

  createBufferSource() {
    const source = new FakeSource()
    FakeAudioContext.sources.push(source)
    return source
  }

  async resume() {}
  async close() {}
}

const callbacks = () => ({
  onLevelChange: vi.fn(),
  onPlayingChange: vi.fn(),
  onResponseCompleted: vi.fn(),
  onSegmentCompleted: vi.fn(),
})

afterEach(() => {
  FakeAudioContext.sources = []
  FakeAudioContext.instance = null
  vi.unstubAllGlobals()
})

describe('SpeakerPcmPlayback', () => {
  it('acknowledges a segment only after all scheduled audio ends', async () => {
    vi.stubGlobal('window', { AudioContext: FakeAudioContext })
    const handlers = callbacks()
    const playback = new SpeakerPcmPlayback(0.5, handlers)
    playback.beginSegment(2, '2:1', 'Hello.')
    await playback.pushPcm16(2, new Int16Array([1, 2, 3]).buffer)
    playback.endSegment(2, '2:1')
    playback.endResponse(2)

    expect(handlers.onSegmentCompleted).not.toHaveBeenCalled()
    expect(handlers.onResponseCompleted).not.toHaveBeenCalled()
    FakeAudioContext.sources[0]?.finish()
    expect(handlers.onSegmentCompleted).toHaveBeenCalledWith(2, '2:1', 'Hello.')
    expect(handlers.onResponseCompleted).toHaveBeenCalledWith(2)
  })

  it('does not acknowledge stopped audio after barge-in', async () => {
    vi.stubGlobal('window', { AudioContext: FakeAudioContext })
    const handlers = callbacks()
    const playback = new SpeakerPcmPlayback(0.5, handlers)
    playback.beginSegment(3, '3:1', 'Interrupted.')
    await playback.pushPcm16(3, new Int16Array([1, 2, 3]).buffer)
    playback.endSegment(3, '3:1')
    playback.clear()
    FakeAudioContext.sources[0]?.finish()

    expect(handlers.onSegmentCompleted).not.toHaveBeenCalled()
  })

  it('schedules concurrently received chunks in strict order', async () => {
    vi.stubGlobal('window', { AudioContext: FakeAudioContext })
    const handlers = callbacks()
    const playback = new SpeakerPcmPlayback(0.5, handlers)
    const chunk = new Int16Array(2_400).buffer
    playback.beginSegment(4, '4:1', 'Ordered audio.')

    await Promise.all([
      playback.pushPcm16(4, chunk),
      playback.pushPcm16(4, chunk),
    ])

    expect(FakeAudioContext.sources).toHaveLength(2)
    expect(FakeAudioContext.sources[0]?.startTime).toBeCloseTo(0.08)
    expect(FakeAudioContext.sources[1]?.startTime).toBeCloseTo(0.18)
  })

  it('does not complete a segment while its chunks are queued for scheduling', async () => {
    vi.stubGlobal('window', { AudioContext: FakeAudioContext })
    const handlers = callbacks()
    const playback = new SpeakerPcmPlayback(0.5, handlers)
    playback.beginSegment(6, '6:1', 'Queued audio.')

    const scheduling = playback.pushPcm16(
      6,
      new Int16Array(2_400).buffer
    )
    playback.endSegment(6, '6:1')
    playback.endResponse(6)

    expect(handlers.onSegmentCompleted).not.toHaveBeenCalled()
    await scheduling
    expect(handlers.onSegmentCompleted).not.toHaveBeenCalled()
    FakeAudioContext.sources[0]?.finish()
    expect(handlers.onSegmentCompleted).toHaveBeenCalledWith(
      6,
      '6:1',
      'Queued audio.'
    )
  })

  it('re-buffers and reports when playback has already underrun', async () => {
    vi.stubGlobal('window', { AudioContext: FakeAudioContext })
    const handlers = {
      ...callbacks(),
      onDiagnostic: vi.fn(),
    }
    const playback = new SpeakerPcmPlayback(0.5, handlers)
    const chunk = new Int16Array(2_400).buffer
    playback.beginSegment(5, '5:1', 'Continuous audio.')
    await playback.pushPcm16(5, chunk)
    if (FakeAudioContext.instance) {
      FakeAudioContext.instance.currentTime = 0.25
    }

    await playback.pushPcm16(5, chunk)

    expect(FakeAudioContext.sources[1]?.startTime).toBeCloseTo(0.33)
    expect(handlers.onDiagnostic).toHaveBeenCalledWith(
      expect.objectContaining({
        activity: 'playback_buffer_underrun',
        generation: 5,
        segmentId: '5:1',
      })
    )
  })
})
