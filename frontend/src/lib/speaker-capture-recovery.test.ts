import { describe, expect, it } from 'vitest'
import {
  SpeakerRecoveryWindow,
  speakerRecoveryWindowMilliseconds,
} from '@/src/lib/speaker-capture-recovery'

describe('SpeakerRecoveryWindow', () => {
  it('exhausts after three failures inside sixty seconds', () => {
    const recovery = new SpeakerRecoveryWindow()

    expect(recovery.record(0)).toEqual({ count: 1, exhausted: false })
    expect(recovery.record(10_000)).toEqual({ count: 2, exhausted: false })
    expect(recovery.record(20_000)).toEqual({ count: 3, exhausted: true })
  })

  it('drops failures outside the rolling window and can be manually reset', () => {
    const recovery = new SpeakerRecoveryWindow()
    recovery.record(0)
    recovery.record(1_000)

    const outsideWindow = speakerRecoveryWindowMilliseconds + 1_001
    expect(recovery.record(outsideWindow)).toEqual({
      count: 1,
      exhausted: false,
    })
    recovery.reset()
    expect(recovery.record(outsideWindow + 1)).toEqual({
      count: 1,
      exhausted: false,
    })
  })
})
