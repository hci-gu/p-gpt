import type { SpeakerAudioEvent } from '@/src/lib/speaker-audio'
import {
  getSpeakerVadActivationConfig,
  speakerVadConfig,
  type SpeakerVadConfig,
  type SpeakerVadDetectionProfile,
} from '@/src/lib/speaker-vad-config'

export const speakerMaximumFrames = Math.floor(60_000 / 32)

type VadStateEvent = (
  | Exclude<SpeakerAudioEvent, { type: 'vad-diagnostic' }>
  | { active: boolean; type: 'activity-change' }
) & { diagnosticDetail?: string }

export class SpeakerVadStateMachine {
  private readonly config: SpeakerVadConfig
  private active = false
  private capped = false
  private candidateFrames = 0
  private candidatePositiveFrames = 0
  private candidateProfile: SpeakerVadDetectionProfile | null = null
  private silentFrames = 0
  private streamedFrames = 0
  private preRoll: Float32Array[] = []

  constructor(config: SpeakerVadConfig = speakerVadConfig) {
    this.config = config
  }

  process(
    audio: Float32Array,
    probability: number,
    profile: SpeakerVadDetectionProfile = 'start'
  ): VadStateEvent[] {
    const events: VadStateEvent[] = []

    if (!this.active) {
      this.preRoll.push(audio)
      this.preRoll = this.preRoll.slice(-this.config.preRollFrames)

      if (this.candidateProfile && this.candidateProfile !== profile) {
        events.push({
          diagnosticDetail: `reason=profile_changed;frames=${this.candidateFrames};positive=${this.candidatePositiveFrames}`,
          type: 'speech-candidate-cancelled',
        })
        this.resetCandidate()
      }

      const activation = getSpeakerVadActivationConfig(this.config, profile)
      if (this.candidateFrames === 0) {
        if (probability < activation.threshold) {
          return events
        }
        this.candidateProfile = profile
        this.candidateFrames = 1
        this.candidatePositiveFrames = 1
        events.push({
          diagnosticDetail: `threshold=${activation.threshold};required=${activation.requiredFrames};window=${activation.windowFrames}`,
          type: 'speech-candidate',
        })
      } else {
        this.candidateFrames += 1
        if (probability >= activation.threshold) {
          this.candidatePositiveFrames += 1
        }
      }

      if (this.candidatePositiveFrames < activation.requiredFrames) {
        if (this.candidateFrames >= activation.windowFrames) {
          events.push({
            diagnosticDetail: `reason=window_exhausted;frames=${this.candidateFrames};positive=${this.candidatePositiveFrames}`,
            type: 'speech-candidate-cancelled',
          })
          this.resetCandidate()
        }
        return events
      }

      this.active = true
      this.capped = false
      const confirmationDetail = `frames=${this.candidateFrames};positive=${this.candidatePositiveFrames};profile=${profile}`
      this.resetCandidate()
      this.silentFrames = 0
      this.streamedFrames = this.preRoll.length
      events.push({ active: true, type: 'activity-change' })
      events.push({ diagnosticDetail: confirmationDetail, type: 'speech-start' })
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

    if (probability >= this.config.continueThreshold) {
      this.silentFrames = 0
      return events
    }

    this.silentFrames += 1
    if (this.silentFrames < this.config.softEndFrames) {
      return events
    }

    const wasCapped = this.capped
    const softEndDetail = `silent_frames=${this.silentFrames};threshold=${this.config.continueThreshold}`
    this.active = false
    this.capped = false
    this.silentFrames = 0
    this.streamedFrames = 0
    this.preRoll = []
    if (!wasCapped) {
      events.push({ diagnosticDetail: softEndDetail, type: 'speech-end' })
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
    this.resetCandidate()
    this.silentFrames = 0
    this.streamedFrames = 0
    this.preRoll = []
    return events
  }

  private resetCandidate() {
    this.candidateFrames = 0
    this.candidatePositiveFrames = 0
    this.candidateProfile = null
  }
}
