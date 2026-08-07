import { describe, expect, it } from 'vitest'
import {
  SpeakerVadStateMachine,
  speakerMaximumFrames,
} from '@/src/lib/speaker-vad-state'

const frame = (value: number) => {
  const audio = new Float32Array(512)
  audio[0] = value
  return audio
}

describe('SpeakerVadStateMachine', () => {
  it('confirms four positive frames within six and flushes twenty-four pre-roll frames', () => {
    const state = new SpeakerVadStateMachine()
    const events = []

    for (let index = 0; index < 18; index += 1) {
      events.push(...state.process(frame(index), 0.1))
    }
    for (const [offset, probability] of [0.9, 0.1, 0.9, 0.1, 0.9, 0.9].entries()) {
      events.push(...state.process(frame(18 + offset), probability))
    }

    expect(events.filter((event) => event.type === 'speech-start')).toHaveLength(1)
    const audioFrames = events.filter((event) => event.type === 'audio-frame')
    expect(audioFrames).toHaveLength(24)
    expect(
      audioFrames.map((event) =>
        event.type === 'audio-frame' ? event.audio[0] : -1
      )
    ).toEqual(Array.from({ length: 24 }, (_, index) => index))
  })

  it('waits for a complete activation window before cancelling a candidate', () => {
    const state = new SpeakerVadStateMachine()
    const events = []

    for (const [index, probability] of [0.9, 0.1, 0.1, 0.9, 0.1, 0.1].entries()) {
      events.push(...state.process(frame(index), probability))
    }

    expect(events.filter((event) => event.type === 'speech-candidate')).toHaveLength(1)
    expect(
      events.filter((event) => event.type === 'speech-candidate-cancelled')
    ).toHaveLength(1)
    expect(events.filter((event) => event.type === 'speech-start')).toHaveLength(
      0
    )
  })

  it('uses continuation hysteresis and soft-ends after twelve quiet frames', () => {
    const state = new SpeakerVadStateMachine()
    for (let index = 0; index < 4; index += 1) {
      state.process(frame(index), 0.9)
    }

    for (let index = 0; index < 11; index += 1) {
      expect(state.process(frame(20 + index), 0.1).map((event) => event.type)).not.toContain(
        'speech-end'
      )
    }
    expect(state.process(frame(31), 0.35).map((event) => event.type)).not.toContain(
      'speech-end'
    )

    let finalSilenceEvents: ReturnType<SpeakerVadStateMachine['process']> = []
    for (let index = 0; index < 12; index += 1) {
      finalSilenceEvents = state.process(frame(40 + index), 0.1)
    }
    expect(finalSilenceEvents.map((event) => event.type)).toContain('speech-end')
  })

  it('does not duplicate trailing audio when a turn reopens', () => {
    const state = new SpeakerVadStateMachine()
    for (let index = 0; index < 4; index += 1) {
      state.process(frame(index), 0.9)
    }
    for (let index = 0; index < 12; index += 1) {
      state.process(frame(20 + index), 0.1)
    }

    const reopenedEvents = []
    for (const [offset, probability] of [0.9, 0.1, 0.9, 0.1, 0.9, 0.9].entries()) {
      reopenedEvents.push(...state.process(frame(40 + offset), probability))
    }
    const reopenedAudio = reopenedEvents.filter(
      (event) => event.type === 'audio-frame'
    )
    expect(reopenedAudio).toHaveLength(6)
    expect(
      reopenedAudio.map((event) =>
        event.type === 'audio-frame' ? event.audio[0] : -1
      )
    ).toEqual([40, 41, 42, 43, 44, 45])
  })

  it('uses the separately configurable barge-in activation profile', () => {
    const state = new SpeakerVadStateMachine({
      bargeIn: { requiredFrames: 2, threshold: 0.6, windowFrames: 3 },
      continueThreshold: 0.3,
      preRollFrames: 24,
      softEndFrames: 12,
      start: { requiredFrames: 4, threshold: 0.4, windowFrames: 6 },
    })

    const events = [
      ...state.process(frame(1), 0.7, 'barge-in'),
      ...state.process(frame(2), 0.7, 'barge-in'),
    ]
    expect(events.map((event) => event.type)).toContain('speech-start')
  })

  it('emits the input limit once and waits for silence before rearming', () => {
    const state = new SpeakerVadStateMachine()
    const emitted = []
    for (let index = 0; index < 4; index += 1) {
      emitted.push(...state.process(frame(index), 0.9))
    }
    for (let index = 4; index < speakerMaximumFrames; index += 1) {
      emitted.push(...state.process(frame(index), 0.9))
    }
    expect(emitted.filter((event) => event.type === 'input-limit')).toHaveLength(1)
    expect(state.process(frame(1), 0.9)).toEqual([])
  })
})
