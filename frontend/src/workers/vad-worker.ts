import { AutoModel, Tensor } from '@huggingface/transformers'
import {
  speakerAudioSampleRate,
  speakerVadFrameSamples,
} from '@/src/lib/speaker-audio'
import type { SpeakerVadDetectionProfile } from '@/src/lib/speaker-vad-config'
import { SpeakerVadStateMachine } from '@/src/lib/speaker-vad-state'

type VadWorkerRequest =
  | { epoch: number; type: 'init' }
  | {
      audio: Float32Array
      capturedAt: number
      epoch: number
      profile: SpeakerVadDetectionProfile
      sequence: number
      type: 'process'
    }
  | { epoch: number; type: 'reset' }

type VadWorkerResponse =
  | { epoch: number; type: 'ready' }
  | { epoch: number; type: 'reset-complete' }
  | { active: boolean; epoch: number; sequence: number; type: 'activity-change' }
  | { diagnosticDetail?: string; epoch: number; sequence: number; type: 'speech-start' }
  | { diagnosticDetail?: string; epoch: number; sequence: number; type: 'speech-candidate' }
  | { diagnosticDetail?: string; epoch: number; sequence: number; type: 'speech-candidate-cancelled' }
  | { audio: Float32Array; epoch: number; sequence: number; type: 'audio-frame' }
  | {
      epoch: number
      probability: number
      processingMilliseconds: number
      queueDelayMilliseconds: number
      sequence: number
      type: 'speech-probability'
    }
  | { diagnosticDetail?: string; epoch: number; sequence: number; type: 'speech-end' }
  | { epoch: number; sequence: number; type: 'input-limit' }
  | { epoch: number; error: string; type: 'error' }

type VadModel = {
  (inputs: {
    input: Tensor
    sr: Tensor
    state: Tensor
  }): Promise<Record<string, Tensor>>
  dispose: () => Promise<unknown[]>
}

const modelId = 'BricksDisplay/silero-vad-6.2'
const modelRevision = '9d91cc0598804f0c00bd8cde231aa86595ec707d'
const contextSamples = 64
const inferenceTimeoutMilliseconds = 5_000

let modelPromise: Promise<VadModel> | null = null
let model: VadModel | null = null
let state = new Float32Array(2 * 128)
let context = new Float32Array(contextSamples)
let vadState = new SpeakerVadStateMachine()
let processingPromise: Promise<void> = Promise.resolve()
let currentEpoch = 0

const postWorkerMessage = (message: VadWorkerResponse) => {
  if (message.type === 'audio-frame') {
    const workerScope = self as unknown as {
      postMessage: (message: unknown, transfer: Transferable[]) => void
    }
    workerScope.postMessage(message, [
      message.audio.buffer as ArrayBuffer,
    ])
    return
  }

  self.postMessage(message)
}

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Voice activity detection failed.'

const reset = () => {
  vadState.reset()
  vadState = new SpeakerVadStateMachine()
  state = new Float32Array(2 * 128)
  context = new Float32Array(contextSamples)
}

const loadModel = async () => {
  if (!modelPromise) {
    modelPromise = AutoModel.from_pretrained(modelId, {
      device: 'wasm',
      dtype: 'q8',
      revision: modelRevision,
    }).then((loadedModel) => {
      model = loadedModel as VadModel
      return model
    })
  }

  return modelPromise
}

const getOutputTensor = (
  output: Record<string, Tensor>,
  preferredKey: string
) => {
  if (output[preferredKey]) {
    return output[preferredKey]
  }

  const fallback = Object.values(output).find((tensor) => tensor instanceof Tensor)
  if (!fallback) {
    throw new Error(`Silero VAD did not return ${preferredKey}.`)
  }

  return fallback
}

const evaluateSpeechProbability = async (audio: Float32Array) => {
  if (audio.length !== speakerVadFrameSamples) {
    throw new Error('Silero VAD frames must contain exactly 512 samples.')
  }

  const loadedModel = await loadModel()
  const input = new Float32Array(contextSamples + audio.length)
  input.set(context)
  input.set(audio, contextSamples)

  const output = await loadedModel({
    input: new Tensor('float32', input, [1, input.length]),
    sr: new Tensor('int64', new BigInt64Array([BigInt(speakerAudioSampleRate)]), []),
    state: new Tensor('float32', state, [2, 1, 128]),
  })
  const probabilityTensor = getOutputTensor(output, 'output')
  const nextStateTensor = output.stateN ?? output.state

  if (!nextStateTensor) {
    throw new Error('Silero VAD did not return its next recurrent state.')
  }

  state = new Float32Array(nextStateTensor.data as Float32Array)
  context = audio.slice(audio.length - contextSamples)

  return Number(probabilityTensor.data[0] ?? 0)
}

const withInferenceTimeout = <Value>(promise: Promise<Value>) =>
  new Promise<Value>((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('Silero VAD inference timed out after 5 seconds.')),
      inferenceTimeoutMilliseconds
    )
    promise.then(
      (value) => {
        clearTimeout(timeout)
        resolve(value)
      },
      (error: unknown) => {
        clearTimeout(timeout)
        reject(error)
      }
    )
  })

const processAudioFrame = async (
  message: Extract<VadWorkerRequest, { type: 'process' }>,
  queuedAt: number
) => {
  if (message.epoch !== currentEpoch) {
    return
  }
  const processingStartedAt = performance.now()
  const probability = await withInferenceTimeout(
    evaluateSpeechProbability(message.audio)
  )
  if (message.epoch !== currentEpoch) {
    return
  }
  const processingMilliseconds = performance.now() - processingStartedAt
  postWorkerMessage({
    epoch: message.epoch,
    probability,
    processingMilliseconds,
    queueDelayMilliseconds: Math.max(0, processingStartedAt - queuedAt),
    sequence: message.sequence,
    type: 'speech-probability',
  })
  for (const event of vadState.process(message.audio, probability, message.profile)) {
    postWorkerMessage({
      ...event,
      epoch: message.epoch,
      sequence: message.sequence,
    })
  }
}

self.addEventListener('message', (event: MessageEvent<VadWorkerRequest>) => {
  const message = event.data

  if (message.type === 'reset') {
    currentEpoch = message.epoch
    processingPromise = processingPromise
      .then(() => {
        reset()
        if (message.epoch === currentEpoch) {
          postWorkerMessage({ epoch: message.epoch, type: 'reset-complete' })
        }
      })
      .catch((error) => {
        postWorkerMessage({
          epoch: message.epoch,
          error: getErrorMessage(error),
          type: 'error',
        })
      })
    return
  }

  if (message.type === 'init') {
    currentEpoch = message.epoch
    const resetBarrier = processingPromise
    void Promise.all([loadModel(), resetBarrier])
      .then(() => {
        if (message.epoch === currentEpoch) {
          postWorkerMessage({ epoch: message.epoch, type: 'ready' })
        }
      })
      .catch((error) =>
        postWorkerMessage({
          epoch: message.epoch,
          error: getErrorMessage(error),
          type: 'error',
        })
      )
    return
  }

  const queuedAt = performance.now()
  processingPromise = processingPromise
    .then(() => processAudioFrame(message, queuedAt))
    .catch((error) => {
      postWorkerMessage({
        epoch: message.epoch,
        error: getErrorMessage(error),
        type: 'error',
      })
    })
})

self.addEventListener('close', () => {
  void model?.dispose()
})
