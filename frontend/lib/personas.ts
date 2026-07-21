export type PersonaRecord = {
  id: string
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
import { pb } from '@/lib/pocketbase'
import type { RecordModel } from 'pocketbase'
