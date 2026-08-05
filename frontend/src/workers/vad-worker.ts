import { AutoModel, Tensor } from '@huggingface/transformers'
import {
  speakerAudioSampleRate,
  speakerVadFrameSamples,
} from '@/src/lib/speaker-audio'
import { SpeakerVadStateMachine } from '@/src/lib/speaker-vad-state'

type VadWorkerRequest =
  | { type: 'init' }
  | { audio: Float32Array; type: 'process' }
  | { type: 'reset' }

type VadWorkerResponse =
  | { type: 'ready' }
  | { active: boolean; type: 'activity-change' }
  | { type: 'speech-start' }
  | { type: 'speech-candidate' }
  | { type: 'speech-candidate-cancelled' }
  | { audio: Float32Array; type: 'audio-frame' }
  | { probability: number; type: 'speech-probability' }
  | { type: 'speech-end' }
  | { type: 'input-limit' }
  | { error: string; type: 'error' }

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

let modelPromise: Promise<VadModel> | null = null
let model: VadModel | null = null
let state = new Float32Array(2 * 128)
let context = new Float32Array(contextSamples)
let vadState = new SpeakerVadStateMachine()
let processingPromise: Promise<void> = Promise.resolve()

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
  for (const event of vadState.reset()) {
    postWorkerMessage(event)
  }
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

const processAudioFrame = async (audio: Float32Array) => {
  const probability = await evaluateSpeechProbability(audio)
  postWorkerMessage({ probability, type: 'speech-probability' })
  for (const event of vadState.process(audio, probability)) {
    postWorkerMessage(event)
  }
}

self.addEventListener('message', (event: MessageEvent<VadWorkerRequest>) => {
  const message = event.data

  if (message.type === 'reset') {
    processingPromise = processingPromise
      .then(() => reset())
      .catch((error) => {
        postWorkerMessage({ error: getErrorMessage(error), type: 'error' })
      })
    return
  }

  if (message.type === 'init') {
    void loadModel()
      .then(() => postWorkerMessage({ type: 'ready' }))
      .catch((error) => postWorkerMessage({ error: getErrorMessage(error), type: 'error' }))
    return
  }

  processingPromise = processingPromise
    .then(() => processAudioFrame(message.audio))
    .catch((error) => {
      postWorkerMessage({ error: getErrorMessage(error), type: 'error' })
    })
})

self.addEventListener('close', () => {
  void model?.dispose()
})
