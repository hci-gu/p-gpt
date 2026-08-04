export type PersonaRecord = {
  id: string
  isDefault: boolean
  name: string
  description: string
  instructionPrompt: string
  profilePicture: string
  profilePictureUrl: string | null
  audioSample: string
  audioSampleUrl: string | null
  created: string
  updated: string
}

export type CreatePersonaInput = {
  name: string
  description: string
  instructionPrompt: string
  profilePicture?: File | null
  audioSample?: File | null
}

export type UpdatePersonaInput = CreatePersonaInput

export type PersonaPreparationInput = {
  personaId: string
  name: string
  instructionPrompt: string
  previousAudioSampleUrl?: string | null
  audioSampleUrl?: string | null
  prepareSystemPrompt: boolean
  prepareVoiceClonePrompt: boolean
}

export type PersonaPreparation = {
  id: string
  status: 'pending' | 'ready' | 'error'
  error: string | null
}

const apiEndpoint = (import.meta.env.VITE_API_ENDPOINT ?? 'http://127.0.0.1:8000').replace(
  /\/$/,
  ''
)

const parsePersona = (value: unknown): PersonaRecord | null => {
  if (
    typeof value !== 'object' ||
    value === null ||
    !('id' in value) ||
    typeof value.id !== 'string' ||
    !('name' in value) ||
    typeof value.name !== 'string'
  ) {
    return null
  }

  const profilePicture =
    'profile_picture' in value && typeof value.profile_picture === 'string'
      ? value.profile_picture
      : ''
  const audioSample =
    'audio_sample' in value && typeof value.audio_sample === 'string'
      ? value.audio_sample
      : ''
  const record = value as unknown as RecordModel

  return {
    id: value.id,
    isDefault:
      !('owner' in value) ||
      typeof value.owner !== 'string' ||
      value.owner.length === 0,
    name: value.name,
    description:
      'description' in value && typeof value.description === 'string'
        ? value.description
        : '',
    instructionPrompt:
      'instruction_prompt' in value &&
      typeof value.instruction_prompt === 'string'
        ? value.instruction_prompt
        : '',
    profilePicture,
    profilePictureUrl: profilePicture
      ? pb.files.getURL(record, profilePicture)
      : null,
    audioSample,
    audioSampleUrl: audioSample
      ? pb.files.getURL(record, audioSample)
      : null,
    created:
      'created' in value && typeof value.created === 'string'
        ? value.created
        : '',
    updated:
      'updated' in value && typeof value.updated === 'string'
        ? value.updated
        : '',
  }
}

export const listPersonas = async (signal?: AbortSignal) => {
  const personas = await pb.collection('personas').getFullList({
    batch: 200,
    requestKey: null,
    signal,
    sort: 'name',
  })

  return personas.flatMap((item) => {
    const persona = parsePersona(item)
    return persona ? [persona] : []
  })
}

export const createPersona = async (input: CreatePersonaInput) => {
  const body = new FormData()
  body.set('name', input.name.trim())
  body.set('description', input.description.trim())
  body.set('instruction_prompt', input.instructionPrompt.trim())

  if (input.profilePicture) {
    body.set('profile_picture', input.profilePicture)
  }
  if (input.audioSample) {
    body.set('audio_sample', input.audioSample)
  }

  const response = await pb.collection('personas').create(body)
  const persona = parsePersona(response)
  if (!persona) {
    throw new Error('PocketBase returned an invalid persona record.')
  }

  return persona
}

export const updatePersona = async (
  personaId: string,
  input: UpdatePersonaInput
) => {
  const body = new FormData()
  body.set('name', input.name.trim())
  body.set('description', input.description.trim())
  body.set('instruction_prompt', input.instructionPrompt.trim())

  if (input.profilePicture) {
    body.set('profile_picture', input.profilePicture)
  }
  if (input.audioSample) {
    body.set('audio_sample', input.audioSample)
  }

  const response = await pb.collection('personas').update(personaId, body)
  const persona = parsePersona(response)
  if (!persona) {
    throw new Error('PocketBase returned an invalid persona record.')
  }

  return persona
}

const parsePreparation = (value: unknown): PersonaPreparation => {
  if (
    typeof value !== 'object' ||
    value === null ||
    !('id' in value) ||
    typeof value.id !== 'string' ||
    !('status' in value) ||
    (value.status !== 'pending' && value.status !== 'ready' && value.status !== 'error')
  ) {
    throw new Error('Backend returned an invalid persona preparation status.')
  }

  return {
    id: value.id,
    status: value.status,
    error: 'error' in value && typeof value.error === 'string' ? value.error : null,
  }
}

const getPreparationError = async (response: Response) => {
  try {
    const payload: unknown = await response.json()
    if (
      typeof payload === 'object' &&
      payload !== null &&
      'detail' in payload &&
      typeof payload.detail === 'string'
    ) {
      return payload.detail
    }
  } catch {
    // Fall back to the generic error below.
  }

  return 'Could not prepare persona changes.'
}

export const startPersonaPreparation = async (
  input: PersonaPreparationInput
) => {
  const response = await fetch(`${apiEndpoint}/persona-preparations`, {
    body: JSON.stringify({
      audio_sample_url: input.audioSampleUrl ?? null,
      instruction_prompt: input.instructionPrompt,
      persona_id: input.personaId,
      persona_name: input.name,
      prepare_system_prompt: input.prepareSystemPrompt,
      prepare_voice_clone_prompt: input.prepareVoiceClonePrompt,
      previous_audio_sample_url: input.previousAudioSampleUrl ?? null,
    }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(await getPreparationError(response))
  }

  return parsePreparation(await response.json())
}

export const getPersonaPreparation = async (preparationId: string) => {
  const response = await fetch(
    `${apiEndpoint}/persona-preparations/${encodeURIComponent(preparationId)}`
  )

  if (!response.ok) {
    throw new Error(await getPreparationError(response))
  }

  return parsePreparation(await response.json())
}
import { pb } from '@/lib/pocketbase'
import type { RecordModel } from 'pocketbase'
