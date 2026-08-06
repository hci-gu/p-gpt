import type { SpeakerAudioEvent } from '@/src/lib/speaker-audio'

export const speakerVadThreshold = 0.5
export const speakerStartFrames = 5
export const speakerSoftEndFrames = 8
export const speakerPreRollFrames = 16
export const speakerMaximumFrames = Math.floor(60_000 / 32)

type VadStateEvent =
  | Exclude<SpeakerAudioEvent, { type: 'vad-diagnostic' }>
  | { active: boolean; type: 'activity-change' }

export class SpeakerVadStateMachine {
  private active = false
  private capped = false
  private positiveFrames = 0
  private silentFrames = 0
  private streamedFrames = 0
  private preRoll: Float32Array[] = []

  process(audio: Float32Array, probability: number): VadStateEvent[] {
    const events: VadStateEvent[] = []

    if (!this.active) {
      this.preRoll.push(audio)
      this.preRoll = this.preRoll.slice(-speakerPreRollFrames)

      if (probability < speakerVadThreshold) {
        if (this.positiveFrames > 0) {
          events.push({ type: 'speech-candidate-cancelled' })
        }
        this.positiveFrames = 0
        return events
      }

      if (this.positiveFrames === 0) {
        events.push({ type: 'speech-candidate' })
      }
      this.positiveFrames += 1
      if (this.positiveFrames < speakerStartFrames) {
        return events
      }

      this.active = true
      this.capped = false
      this.positiveFrames = 0
      this.silentFrames = 0
      this.streamedFrames = this.preRoll.length
      events.push({ active: true, type: 'activity-change' })
      events.push({ type: 'speech-start' })
      for (const pendingFrame of this.preRoll) {
        events.push({
          audio: pendingFrame,
          sampleRate: 16_000,
          type: 'audio-frame',
        })
      }
      this.preRoll = []
      return events
    }

    if (!this.capped) {
      events.push({ audio, sampleRate: 16_000, type: 'audio-frame' })
      this.streamedFrames += 1
      if (this.streamedFrames >= speakerMaximumFrames) {
        this.capped = true
        events.push({ type: 'input-limit' })
      }
    }

    if (probability >= speakerVadThreshold) {
      this.silentFrames = 0
      return events
    }

    this.silentFrames += 1
    if (this.silentFrames < speakerSoftEndFrames) {
      return events
    }

    const wasCapped = this.capped
    this.active = false
    this.capped = false
    this.silentFrames = 0
    this.streamedFrames = 0
    this.preRoll = []
    if (!wasCapped) {
      events.push({ type: 'speech-end' })
    }
    events.push({ active: false, type: 'activity-change' })
    return events
  }

  reset(): VadStateEvent[] {
    const events: VadStateEvent[] = []
    if (this.active) {
      if (!this.capped) {
        events.push({ type: 'speech-end' })
      }
      events.push({ active: false, type: 'activity-change' })
    }
    this.active = false
    this.capped = false
    this.positiveFrames = 0
    this.silentFrames = 0
    this.streamedFrames = 0
    this.preRoll = []
    return events
  }
}
