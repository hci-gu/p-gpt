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

const notifyChatHistoryUpdated = () => {
  window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT))
}

export const listChatHistories = async (signal?: AbortSignal) => {
  const records = await pb.collection('chat_history').getFullList({
    batch: 200,
    requestKey: null,
    signal,
  })

  return records.flatMap((item) => {
    const record = parseRecord(item)
    return record ? [record] : []
  })
}

export const createChatHistory = async (
  conversation: StoredChatMessage[],
  personaId: string | null
) => {
  const response = await pb.collection('chat_history').create({
    conversation,
    persona_id: personaId ?? '',
    title: createChatTitle(conversation),
  })
  const record = parseRecord(response)
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
  await pb.collection('chat_history').update(recordId, {
    conversation,
    title: createChatTitle(conversation),
  })

  notifyChatHistoryUpdated()
}

export const deleteChatHistory = async (recordId: string) => {
  await pb.collection('chat_history').delete(recordId)

  notifyChatHistoryUpdated()
}
import { pb } from '@/lib/pocketbase'
