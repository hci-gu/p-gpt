import { afterEach, describe, expect, it, vi } from 'vitest'
import { SpeakerPcmPlayback } from '@/src/lib/speaker-playback'

class FakeSource {
  buffer: { duration: number } | null = null
  onended: (() => void) | null = null

  connect() {}
  start() {}
  stop() {}
  finish() {
    this.onended?.()
  }
}

class FakeAudioContext {
  static sources: FakeSource[] = []
  currentTime = 0
  destination = {}

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
})
