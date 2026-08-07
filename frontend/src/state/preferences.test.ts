import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  defaultGenerationParameters,
  usePreferencesStore,
} from './preferences'

describe('speech language preference', () => {
  const setItem = vi.fn()

  beforeEach(() => {
    setItem.mockClear()
    vi.stubGlobal('window', { localStorage: { setItem } })
  })

  afterEach(() => {
    usePreferencesStore.getState().resetGenerationParameters()
    vi.unstubAllGlobals()
  })

  it('defaults to English', () => {
    expect(defaultGenerationParameters.speechLanguage).toBe('en')
  })

  it('is shared through the preferences store', () => {
    usePreferencesStore
      .getState()
      .setGenerationParameter('speechLanguage', 'sv')

    expect(
      usePreferencesStore.getState().generationParameters.speechLanguage
    ).toBe('sv')
    expect(setItem).toHaveBeenCalledWith(
      'p-gpt-generation-parameters',
      expect.stringContaining('"speechLanguage":"sv"')
    )
  })
})
