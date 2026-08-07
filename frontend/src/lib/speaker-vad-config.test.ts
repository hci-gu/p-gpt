import { describe, expect, it, vi } from 'vitest'
import {
  defaultSpeakerVadConfig,
  parseSpeakerVadConfig,
} from '@/src/lib/speaker-vad-config'

describe('parseSpeakerVadConfig', () => {
  it('parses valid Vite overrides', () => {
    const config = parseSpeakerVadConfig(
      {
        VITE_SPEAKER_VAD_BARGE_REQUIRED_FRAMES: '3',
        VITE_SPEAKER_VAD_BARGE_THRESHOLD: '0.35',
        VITE_SPEAKER_VAD_BARGE_WINDOW_FRAMES: '5',
        VITE_SPEAKER_VAD_CONTINUE_THRESHOLD: '0.25',
        VITE_SPEAKER_VAD_PRE_ROLL_FRAMES: '30',
        VITE_SPEAKER_VAD_SOFT_END_FRAMES: '15',
        VITE_SPEAKER_VAD_START_REQUIRED_FRAMES: '5',
        VITE_SPEAKER_VAD_START_THRESHOLD: '0.45',
        VITE_SPEAKER_VAD_START_WINDOW_FRAMES: '7',
      },
      vi.fn()
    )

    expect(config).toEqual({
      bargeIn: { requiredFrames: 3, threshold: 0.35, windowFrames: 5 },
      continueThreshold: 0.25,
      preRollFrames: 30,
      softEndFrames: 15,
      start: { requiredFrames: 5, threshold: 0.45, windowFrames: 7 },
    })
  })

  it('falls back when required frames exceed the activation window', () => {
    const warn = vi.fn()
    const config = parseSpeakerVadConfig(
      {
        VITE_SPEAKER_VAD_START_REQUIRED_FRAMES: '7',
        VITE_SPEAKER_VAD_START_THRESHOLD: '0.45',
        VITE_SPEAKER_VAD_START_WINDOW_FRAMES: '4',
      },
      warn
    )

    expect(config.start).toEqual(defaultSpeakerVadConfig.start)
    expect(warn).toHaveBeenCalledOnce()
  })

  it('falls back for non-finite, out-of-range, and fractional frame values', () => {
    const warn = vi.fn()
    const config = parseSpeakerVadConfig(
      {
        VITE_SPEAKER_VAD_CONTINUE_THRESHOLD: '2',
        VITE_SPEAKER_VAD_PRE_ROLL_FRAMES: '12.5',
        VITE_SPEAKER_VAD_SOFT_END_FRAMES: 'not-a-number',
      },
      warn
    )

    expect(config.continueThreshold).toBe(defaultSpeakerVadConfig.continueThreshold)
    expect(config.preRollFrames).toBe(defaultSpeakerVadConfig.preRollFrames)
    expect(config.softEndFrames).toBe(defaultSpeakerVadConfig.softEndFrames)
    expect(warn).toHaveBeenCalledTimes(3)
  })
})
