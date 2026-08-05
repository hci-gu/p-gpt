export const speakerWebSocketProtocol = 'p-gpt-speaker.v1'

export type SpeakerStatus =
  | 'connecting'
  | 'listening'
  | 'hearing'
  | 'thinking'
  | 'speaking'
  | 'reconnecting'
  | 'error'
  | 'idle'

export type SpeakerServerEvent = {
  type: string
  eventId: string
  sessionId: string
  turnId?: string
  turnRevision?: number
  responseGeneration?: number
  segmentId?: string
  text?: string
  code?: string
  message?: string
  fatal?: boolean
  finishReason?: 'interrupted'
  reason?: string
  encoding?: 'pcm_s16le'
  sampleRate?: number
}

export const createSpeakerEvent = (
  type: string,
  values: Record<string, unknown> = {}
) => ({
  type,
  eventId: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
  ...values,
})

export const getSpeakerSocketUrl = (apiEndpoint: string) => {
  const url = new URL(apiEndpoint)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = `${url.pathname.replace(/\/$/, '')}/speaker/v1`
  url.search = ''
  url.hash = ''
  return url.toString()
}
