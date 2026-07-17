export type StoredChatMessage = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export type ChatHistoryRecord = {
  id: string
  title: string
  personaId: string | null
  conversation: StoredChatMessage[]
  created: string
  updated: string
}

type PocketBaseListResponse = {
  items?: unknown[]
  page?: number
  totalPages?: number
}

const pocketBaseEndpoint = (
  import.meta.env.VITE_POCKETBASE_ENDPOINT ?? 'http://127.0.0.1:8090'
).replace(/\/$/, '')

const recordsEndpoint = `${pocketBaseEndpoint}/api/collections/chat_history/records`

export const CHAT_HISTORY_UPDATED_EVENT = 'chat-history-updated'

export const createChatTitle = (conversation: StoredChatMessage[]) => {
  const firstUserMessage = conversation.find((message) => message.role === 'user')
  const words = firstUserMessage?.content.trim().split(/\s+/).slice(0, 4) ?? []

  return words.length > 0 ? `${words.join(' ')}...` : 'Untitled chat'
}

const isStoredRole = (role: unknown): role is StoredChatMessage['role'] =>
  role === 'user' || role === 'assistant' || role === 'system'

const parseConversation = (value: unknown): StoredChatMessage[] => {
  const messages =
    Array.isArray(value)
      ? value
      : typeof value === 'object' &&
          value !== null &&
          'messages' in value &&
          Array.isArray(value.messages)
        ? value.messages
        : []

  return messages.flatMap((message) => {
    if (typeof message !== 'object' || message === null) {
      return []
    }

    const role = 'role' in message ? message.role : undefined
    const content = 'content' in message ? message.content : undefined

    if (!isStoredRole(role) || typeof content !== 'string') {
      return []
    }

    return [{ role, content }]
  })
}

const parseRecord = (value: unknown): ChatHistoryRecord | null => {
  if (typeof value !== 'object' || value === null || !('id' in value)) {
    return null
  }

  if (typeof value.id !== 'string') {
    return null
  }

  return {
    id: value.id,
    title:
      'title' in value && typeof value.title === 'string' && value.title.trim()
        ? value.title
        : createChatTitle(
            parseConversation('conversation' in value ? value.conversation : [])
          ),
    personaId:
      'persona_id' in value &&
      typeof value.persona_id === 'string' &&
      value.persona_id
        ? value.persona_id
        : null,
    conversation: parseConversation(
      'conversation' in value ? value.conversation : []
    ),
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

const notifyChatHistoryUpdated = () => {
  window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT))
}

export const listChatHistories = async (signal?: AbortSignal) => {
  const records: ChatHistoryRecord[] = []
  let page = 1
  let totalPages = 1

  do {
    const query = new URLSearchParams({
      page: String(page),
      perPage: '200',
    })
    const response = await fetch(`${recordsEndpoint}?${query}`, {
      headers: { Accept: 'application/json' },
      signal,
    })

    if (!response.ok) {
      await throwResponseError(response)
    }

    const data = (await response.json()) as PocketBaseListResponse
    records.push(
      ...(data.items ?? []).flatMap((item) => {
        const record = parseRecord(item)
        return record ? [record] : []
      })
    )
    totalPages = data.totalPages ?? 1
    page += 1
  } while (page <= totalPages)

  return records
}

export const createChatHistory = async (
  conversation: StoredChatMessage[],
  personaId: string | null
) => {
  const response = await fetch(recordsEndpoint, {
    body: JSON.stringify({
      conversation,
      persona_id: personaId ?? '',
      title: createChatTitle(conversation),
    }),
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    method: 'POST',
  })

  if (!response.ok) {
    await throwResponseError(response)
  }

  const record = parseRecord(await response.json())
  if (!record) {
    throw new Error('PocketBase returned an invalid chat history record.')
  }

  notifyChatHistoryUpdated()
  return record
}

export const updateChatHistory = async (
  recordId: string,
  conversation: StoredChatMessage[]
) => {
  const response = await fetch(`${recordsEndpoint}/${recordId}`, {
    body: JSON.stringify({
      conversation,
      title: createChatTitle(conversation),
    }),
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    method: 'PATCH',
  })

  if (!response.ok) {
    await throwResponseError(response)
  }

  notifyChatHistoryUpdated()
}

export const deleteChatHistory = async (recordId: string) => {
  const response = await fetch(`${recordsEndpoint}/${recordId}`, {
    method: 'DELETE',
  })

  if (!response.ok) {
    await throwResponseError(response)
  }

  notifyChatHistoryUpdated()
}
