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
  it('requires five positive frames and flushes sixteen pre-roll frames', () => {
    const state = new SpeakerVadStateMachine()
    const events = []

    for (let index = 0; index < 11; index += 1) {
      events.push(...state.process(frame(index), 0.1))
    }
    for (let index = 11; index < 16; index += 1) {
      events.push(...state.process(frame(index), 0.9))
    }

    expect(events.filter((event) => event.type === 'speech-start')).toHaveLength(1)
    const audioFrames = events.filter((event) => event.type === 'audio-frame')
    expect(audioFrames).toHaveLength(16)
    expect(
      audioFrames.map((event) =>
        event.type === 'audio-frame' ? event.audio[0] : -1
      )
    ).toEqual(Array.from({ length: 16 }, (_, index) => index))
  })

  it('cancels a candidate that does not reach speech start', () => {
    const state = new SpeakerVadStateMachine()

    expect(state.process(frame(1), 0.9).map((event) => event.type)).toContain(
      'speech-candidate'
    )
    expect(state.process(frame(2), 0.1).map((event) => event.type)).toContain(
      'speech-candidate-cancelled'
    )
  })

  it('soft-ends after eight silent frames and does not duplicate old audio on reopen', () => {
    const state = new SpeakerVadStateMachine()
    for (let index = 0; index < 5; index += 1) {
      state.process(frame(index), 0.9)
    }

    let finalSilenceEvents: ReturnType<SpeakerVadStateMachine['process']> = []
    for (let index = 0; index < 8; index += 1) {
      finalSilenceEvents = state.process(frame(20 + index), 0.1)
    }
    expect(finalSilenceEvents.map((event) => event.type)).toContain('speech-end')

    const reopenedEvents = []
    for (let index = 0; index < 5; index += 1) {
      reopenedEvents.push(...state.process(frame(40 + index), 0.9))
    }
    const reopenedAudio = reopenedEvents.filter(
      (event) => event.type === 'audio-frame'
    )
    expect(reopenedAudio).toHaveLength(5)
    expect(
      reopenedAudio.map((event) =>
        event.type === 'audio-frame' ? event.audio[0] : -1
      )
    ).toEqual([40, 41, 42, 43, 44])
  })

  it('emits the input limit once and waits for silence before rearming', () => {
    const state = new SpeakerVadStateMachine()
    const emitted = []
    for (let index = 0; index < 5; index += 1) {
      emitted.push(...state.process(frame(index), 0.9))
    }
    for (let index = 5; index < speakerMaximumFrames; index += 1) {
      emitted.push(...state.process(frame(index), 0.9))
    }
    expect(emitted.filter((event) => event.type === 'input-limit')).toHaveLength(1)
    expect(state.process(frame(1), 0.9)).toEqual([])
  })
})
