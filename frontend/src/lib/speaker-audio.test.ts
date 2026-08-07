import { describe, expect, it } from 'vitest'
import { float32ToPcm16 } from '@/src/lib/speaker-audio'

describe('float32ToPcm16', () => {
  it('clamps samples and writes signed little-endian PCM', () => {
    const output = float32ToPcm16(
      new Float32Array([-2, -1, -0.5, 0, 0.5, 1, 2])
    )
    const view = new DataView(output)

    expect(
      Array.from({ length: 7 }, (_, index) => view.getInt16(index * 2, true))
    ).toEqual([-32_768, -32_768, -16_384, 0, 16_384, 32_767, 32_767])
  })
})
