import { pipeline } from '@huggingface/transformers'

export type SpeechLanguage = 'en' | 'sv'

export type AsrWorkerRequest =
  | {
      language: SpeechLanguage
      type: 'init'
    }
  | {
      language: SpeechLanguage
      type: 'setLanguage'
    }
  | {
      audio: Float32Array
      isFinal: boolean
      language: SpeechLanguage
      sampleRate: number
      sessionId: string
      type: 'transcribe'
    }

export type AsrWorkerResponse =
  | {
      device: string
      dtype: string
      type: 'ready'
    }
  | {
      isFinal: boolean
      language: SpeechLanguage
      sessionId: string
      text: string
      type: 'partial' | 'final'
    }
  | {
      error: string
      type: 'error'
    }

type AsrDevice = 'webgpu' | 'wasm'
type AsrDtype = 'q4' | 'q8' | 'fp16' | 'fp32'
type TranscribeRequest = Extract<AsrWorkerRequest, { type: 'transcribe' }>
type AsrPipeline = (
  audio: Float32Array,
  options: {
    chunk_length_s?: number
    force_full_sequences?: boolean
    language: SpeechLanguage
    stride_length_s?: number
    task: 'transcribe'
  }
) => Promise<{ text: string } | { text: string }[]>

const modelId = 'onnx-community/whisper-tiny'
const targetSampleRate = 16_000
const dtypeCandidates: AsrDtype[] = ['q4', 'q8', 'fp16', 'fp32']

let currentLanguage: SpeechLanguage = 'en'
let loadPromise: Promise<void> | null = null
let transcriber: AsrPipeline | null = null
let selectedDevice: AsrDevice | null = null
let selectedDtype: AsrDtype | null = null
let activeRequest = false
let pendingRequest: TranscribeRequest | null = null

const postWorkerMessage = (message: AsrWorkerResponse) => {
  self.postMessage(message)
}

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Transcription failed.'

const getDeviceCandidates = (): AsrDevice[] =>
  'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm']

const createPipeline = async () => {
  let lastError: unknown = null

  for (const device of getDeviceCandidates()) {
    for (const dtype of dtypeCandidates) {
      try {
        const candidate = await pipeline(
          'automatic-speech-recognition',
          modelId,
          { device, dtype }
        )

        transcriber = candidate as AsrPipeline
        selectedDevice = device
        selectedDtype = dtype
        return
      } catch (error) {
        lastError = error
      }
    }
  }

  throw lastError ?? new Error('Unable to load the ASR model.')
}

const ensurePipeline = async () => {
  if (!loadPromise) {
    loadPromise = createPipeline().then(() => {
      if (!(selectedDevice && selectedDtype)) {
        throw new Error('ASR model loaded without device metadata.')
      }

      postWorkerMessage({
        device: selectedDevice,
        dtype: selectedDtype,
        type: 'ready',
      })
    })
  }

  await loadPromise
}

const getOutputText = (output: { text: string } | { text: string }[]) => {
  if (Array.isArray(output)) {
    return output.map((item) => item.text).join(' ').trim()
  }

  return output.text.trim()
}

const runTranscription = async (request: TranscribeRequest) => {
  activeRequest = true

  try {
    await ensurePipeline()

    if (!transcriber) {
      throw new Error('ASR model is not ready.')
    }

    if (
      request.sampleRate !== targetSampleRate ||
      request.audio.length < targetSampleRate * 0.35
    ) {
      if (request.isFinal) {
        postWorkerMessage({
          isFinal: true,
          language: request.language,
          sessionId: request.sessionId,
          text: '',
          type: 'final',
        })
      }
      return
    }

    const output = await transcriber(request.audio, {
      chunk_length_s: 15,
      force_full_sequences: false,
      language: request.language,
      stride_length_s: 2,
      task: 'transcribe',
    })

    postWorkerMessage({
      isFinal: request.isFinal,
      language: request.language,
      sessionId: request.sessionId,
      text: getOutputText(output),
      type: request.isFinal ? 'final' : 'partial',
    })
  } catch (error) {
    postWorkerMessage({
      error: getErrorMessage(error),
      type: 'error',
    })
  } finally {
    activeRequest = false

    if (pendingRequest) {
      const nextRequest = pendingRequest
      pendingRequest = null
      void runTranscription(nextRequest)
    }
  }
}

const enqueueTranscription = (request: TranscribeRequest) => {
  if (activeRequest) {
    pendingRequest = request
    return
  }

  void runTranscription(request)
}

self.addEventListener('message', (event: MessageEvent<AsrWorkerRequest>) => {
  const message = event.data

  if (message.type === 'init') {
    currentLanguage = message.language
    void ensurePipeline().catch((error) => {
      postWorkerMessage({
        error: getErrorMessage(error),
        type: 'error',
      })
    })
    return
  }

  if (message.type === 'setLanguage') {
    currentLanguage = message.language
    return
  }

  enqueueTranscription({
    ...message,
    language: message.language ?? currentLanguage,
  })
})
