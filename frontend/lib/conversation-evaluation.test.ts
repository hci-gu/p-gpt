import { describe, expect, it } from 'vitest'
import {
  canEvaluateChatHistory,
} from '@/lib/conversation-evaluation'
import type { ChatHistoryRecord } from '@/lib/chat-history'

const record = (overrides: Partial<ChatHistoryRecord> = {}): ChatHistoryRecord => ({
  completedAt: null,
  conversation: [{ role: 'user', content: 'Hello' }],
  created: '',
  evaluation: null,
  id: 'chat-1',
  personaId: null,
  status: 'active',
  title: 'Practice chat',
  updated: '',
  ...overrides,
})

describe('canEvaluateChatHistory', () => {
  it('allows a saved, non-empty active chat when idle', () => {
    expect(canEvaluateChatHistory(record(), 'chat-1', 'ready')).toBe(true)
  })

  it('hides evaluation for empty, generating, and completed chats', () => {
    expect(canEvaluateChatHistory(record({ conversation: [] }), null, 'ready')).toBe(false)
    expect(canEvaluateChatHistory(record(), 'chat-1', 'streaming')).toBe(false)
    expect(canEvaluateChatHistory(record({ status: 'completed' }), null, 'ready')).toBe(false)
  })
})
