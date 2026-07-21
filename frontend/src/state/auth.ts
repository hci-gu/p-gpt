import { pb } from '@/lib/pocketbase'
import { useChatStore } from '@/src/state/chat'
import { usePersonasStore } from '@/src/state/personas'
import type { RecordModel } from 'pocketbase'
import { create } from 'zustand'

type AuthStatus = 'loading' | 'authenticated' | 'anonymous'

interface AuthState {
  user: RecordModel | null
  status: AuthStatus
  initialize: () => Promise<void>
  signIn: (email: string, password: string) => Promise<void>
  signOut: () => void
  signUp: (
    email: string,
    password: string,
    passwordConfirm: string
  ) => Promise<void>
}

let initialized = false
let initializationPromise: Promise<void> | null = null
let currentUserId = pb.authStore.record?.id ?? null

const clearUserSpecificState = () => {
  pb.cancelAllRequests()
  useChatStore.getState().resetForAuthChange()
  usePersonasStore.getState().resetForAuthChange()
}

export const useAuthStore = create<AuthState>((set) => {
  const syncAuthState = () => {
    const user = pb.authStore.isValid ? pb.authStore.record : null
    set({
      status: user ? 'authenticated' : 'anonymous',
      user,
    })
  }

  pb.authStore.onChange((_token, record) => {
    const nextUserId = record?.id ?? null
    if (nextUserId !== currentUserId) {
      clearUserSpecificState()
      currentUserId = nextUserId
    }

    if (initialized) {
      syncAuthState()
    }
  })

  return {
    user: null,
    status: 'loading',
    initialize: async () => {
      if (!initializationPromise) {
        initializationPromise = (async () => {
          if (pb.authStore.isValid) {
            try {
              await pb.collection('users').authRefresh()
            } catch {
              pb.authStore.clear()
            }
          } else if (pb.authStore.record || pb.authStore.token) {
            pb.authStore.clear()
          }

          initialized = true
          syncAuthState()
        })()
      }

      await initializationPromise
    },
    signIn: async (email, password) => {
      await pb.collection('users').authWithPassword(email.trim(), password)
    },
    signOut: () => {
      pb.authStore.clear()
    },
    signUp: async (email, password, passwordConfirm) => {
      const identity = email.trim()
      await pb.collection('users').create({
        email: identity,
        emailVisibility: false,
        password,
        passwordConfirm,
      })
      await pb.collection('users').authWithPassword(identity, password)
    },
  }
})
