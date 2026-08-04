export const asrModels = ['tiny', 'base', 'small'] as const

export type AsrModel = (typeof asrModels)[number]

export const defaultAsrModel: AsrModel = 'tiny'

export const asrModelIds: Record<AsrModel, string> = {
  tiny: 'onnx-community/whisper-tiny',
  base: 'onnx-community/whisper-base',
  small: 'onnx-community/whisper-small',
}
