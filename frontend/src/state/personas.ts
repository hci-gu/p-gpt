import {
  createPersona as createPersonaRecord,
  type CreatePersonaInput,
  listPersonas,
  type PersonaRecord,
} from '@/lib/personas'
import { create } from 'zustand'

const personaStorageKey = 'p-gpt-persona'

const getStoredPersonaId = () => {
  try {
    return window.localStorage.getItem(personaStorageKey)
  } catch {
    return null
  }
}

let personasLoadPromise: Promise<PersonaRecord[]> | null = null

interface PersonasState {
  personas: PersonaRecord[]
  selectedPersonaId: string | null
  isLoading: boolean
  hasLoaded: boolean
  loadError: string | null
  ensurePersonasLoaded: () => Promise<PersonaRecord[]>
  createPersona: (input: CreatePersonaInput) => Promise<PersonaRecord>
  selectPersona: (personaId: string) => void
}

export const usePersonasStore = create<PersonasState>((set, get) => ({
  personas: [],
  selectedPersonaId: getStoredPersonaId(),
  isLoading: false,
  hasLoaded: false,
  loadError: null,
  ensurePersonasLoaded: async () => {
    if (get().hasLoaded) {
      return get().personas
    }

    if (!personasLoadPromise) {
      set({ isLoading: true, loadError: null })
      personasLoadPromise = listPersonas()
    }

    try {
      const personas = await personasLoadPromise
      const storedPersonaId = get().selectedPersonaId
      const selectedPersonaId = personas.some(
        (persona) => persona.id === storedPersonaId
      )
        ? storedPersonaId
        : (personas[0]?.id ?? null)

      if (selectedPersonaId) {
        try {
          window.localStorage.setItem(personaStorageKey, selectedPersonaId)
        } catch {
          // The in-memory selection still applies when storage is unavailable.
        }
      }

      set({
        hasLoaded: true,
        isLoading: false,
        loadError: null,
        personas,
        selectedPersonaId,
      })
      return personas
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Could not load personas.'
      set({ isLoading: false, loadError: message })
      personasLoadPromise = null
      throw error
    }
  },
  createPersona: async (input) => {
    const persona = await createPersonaRecord(input)
    set((state) => ({
      hasLoaded: true,
      personas: [...state.personas, persona].sort((first, second) =>
        first.name.localeCompare(second.name)
      ),
    }))
    get().selectPersona(persona.id)
    return persona
  },
  selectPersona: (personaId) => {
    try {
      window.localStorage.setItem(personaStorageKey, personaId)
    } catch {
      // The in-memory selection still applies when storage is unavailable.
    }

    set({ selectedPersonaId: personaId })
  },
}))
