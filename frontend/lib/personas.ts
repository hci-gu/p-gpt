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

type PocketBaseListResponse = {
  items?: unknown[]
  totalPages?: number
}

const pocketBaseEndpoint = (
  import.meta.env.VITE_POCKETBASE_ENDPOINT ?? 'http://127.0.0.1:8090'
).replace(/\/$/, '')

const collectionName = 'personas'
const recordsEndpoint = `${pocketBaseEndpoint}/api/collections/${collectionName}/records`

const getFileUrl = (recordId: string, filename: string) =>
  filename
    ? `${pocketBaseEndpoint}/api/files/${collectionName}/${encodeURIComponent(
        recordId
      )}/${encodeURIComponent(filename)}`
    : null

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
    profilePictureUrl: getFileUrl(value.id, profilePicture),
    audioSample,
    audioSampleUrl: getFileUrl(value.id, audioSample),
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

const throwResponseError = async (response: Response) => {
  let message = `PocketBase request failed with status ${response.status}`

  try {
    const data: unknown = await response.json()
    if (
      typeof data === 'object' &&
      data !== null &&
      'message' in data &&
      typeof data.message === 'string'
    ) {
      message = data.message
    }
  } catch {
    // PocketBase may return an empty response body for some failures.
  }

  throw new Error(message)
}

export const listPersonas = async (signal?: AbortSignal) => {
  const personas: PersonaRecord[] = []
  let page = 1
  let totalPages = 1

  do {
    const query = new URLSearchParams({
      page: String(page),
      perPage: '200',
      sort: 'name',
    })
    const response = await fetch(`${recordsEndpoint}?${query}`, {
      headers: { Accept: 'application/json' },
      signal,
    })

    if (!response.ok) {
      await throwResponseError(response)
    }

    const data = (await response.json()) as PocketBaseListResponse
    personas.push(
      ...(data.items ?? []).flatMap((item) => {
        const persona = parsePersona(item)
        return persona ? [persona] : []
      })
    )
    totalPages = data.totalPages ?? 1
    page += 1
  } while (page <= totalPages)

  return personas
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

  const response = await fetch(recordsEndpoint, {
    body,
    headers: { Accept: 'application/json' },
    method: 'POST',
  })

  if (!response.ok) {
    await throwResponseError(response)
  }

  const persona = parsePersona(await response.json())
  if (!persona) {
    throw new Error('PocketBase returned an invalid persona record.')
  }

  return persona
}
