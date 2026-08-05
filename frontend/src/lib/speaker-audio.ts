export const speakerAudioSampleRate = 16_000
export const speakerVadFrameSamples = 512
export const speakerInputFrameBytes = speakerVadFrameSamples * 2
export const speakerOutputSampleRate = 24_000

export const float32ToPcm16 = (audio: Float32Array) => {
  const output = new ArrayBuffer(audio.length * 2)
  const view = new DataView(output)

  for (let index = 0; index < audio.length; index += 1) {
    const sample = Math.min(1, Math.max(-1, audio[index] ?? 0))
    const pcm = sample < 0 ? Math.round(sample * 32_768) : Math.round(sample * 32_767)
    view.setInt16(index * 2, pcm, true)
  }

  return output
}

export type SpeakerAudioEvent =
  | {
      type: 'speech-candidate'
    }
  | {
      type: 'speech-candidate-cancelled'
    }
  | {
      type: 'speech-start'
    }
  | {
      audio: Float32Array
      sampleRate: typeof speakerAudioSampleRate
      type: 'audio-frame'
    }
  | {
      type: 'speech-end'
    }
  | {
      type: 'input-limit'
    }

export type SpeakerAudioConsumer = (event: SpeakerAudioEvent) => void

export const discardSpeakerAudioEvent: SpeakerAudioConsumer = () => {
  // Backend audio transport is intentionally added in a later change.
}
