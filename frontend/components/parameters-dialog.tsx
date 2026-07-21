import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import {
  defaultGenerationParameters,
  omnivoiceNumStepsFromLevel,
  usePreferencesStore,
} from '@/src/state/preferences'

type ParametersDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const minTokenExponent = 6
const maxTokenExponent = 13
const tokenCheckpoints = [128, 256, 512, 1024, 2048, 4096]
const tokenSnapDistance = 0.13

const tokensToExponent = (tokens: number) => Math.log2(tokens)

const exponentToTokens = (exponent: number) => {
  const closestCheckpoint = tokenCheckpoints.reduce((closest, checkpoint) =>
    Math.abs(tokensToExponent(checkpoint) - exponent) <
    Math.abs(tokensToExponent(closest) - exponent)
      ? checkpoint
      : closest
  )

  if (
    Math.abs(tokensToExponent(closestCheckpoint) - exponent) <=
    tokenSnapDistance
  ) {
    return closestCheckpoint
  }

  return Math.min(8192, Math.max(64, Math.round(2 ** exponent)))
}

const checkpointPosition = (checkpoint: number) =>
  `${
    ((tokensToExponent(checkpoint) - minTokenExponent) /
      (maxTokenExponent - minTokenExponent)) *
    100
  }%`

export function ParametersDialog({
  open,
  onOpenChange,
}: ParametersDialogProps) {
  const parameters = usePreferencesStore((state) => state.generationParameters)
  const setParameter = usePreferencesStore(
    (state) => state.setGenerationParameter
  )
  const resetParameters = usePreferencesStore(
    (state) => state.resetGenerationParameters
  )

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Parameters</DialogTitle>
          <DialogDescription>
            Tune text generation and speech behavior for future responses.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3">
          <section className="grid gap-3 rounded-lg border bg-muted/15 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="font-medium">Temperature</h3>
                <p className="text-muted-foreground">
                  Higher values make responses more varied and unpredictable.
                </p>
              </div>
              <output className="min-w-12 rounded-md border bg-background px-2 py-1 text-center font-mono tabular-nums">
                {parameters.temperature.toFixed(2)}
              </output>
            </div>
            <Slider
              aria-label="Temperature"
              max={2}
              min={0}
              onValueChange={([value]) =>
                setParameter('temperature', Number(value.toFixed(2)))
              }
              step={0.01}
              value={[parameters.temperature]}
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>0.00</span>
              <span>1.00</span>
              <span>2.00</span>
            </div>
          </section>

          <div className="grid gap-3 sm:grid-cols-2">
            <section className="grid content-start gap-3 rounded-lg border bg-muted/15 p-4">
              <div>
                <h3 className="font-medium">Seed</h3>
                <p className="text-muted-foreground">
                  Use an integer for repeatable output, or leave empty for random.
                </p>
              </div>
              <Input
                aria-label="Generation seed"
                inputMode="numeric"
                min={0}
                onChange={(event) => {
                  if (event.target.value === '') {
                    setParameter('seed', null)
                    return
                  }

                  const seed = Number(event.target.value)
                  if (Number.isSafeInteger(seed) && seed >= 0) {
                    setParameter('seed', seed)
                  }
                }}
                placeholder="Random"
                step={1}
                type="number"
                value={parameters.seed ?? ''}
              />
            </section>

            <section className="grid content-start gap-3 rounded-lg border bg-muted/15 p-4">
              <div>
                <h3 className="font-medium">Repeat penalty</h3>
                <p className="text-muted-foreground">
                  Discourages the model from repeating text.
                </p>
              </div>
              <Select
                onValueChange={(value) => {
                  const repeatPenalty = Number(value)
                  if (
                    repeatPenalty === 1 ||
                    repeatPenalty === 1.1 ||
                    repeatPenalty === 1.2
                  ) {
                    setParameter('repeatPenalty', repeatPenalty)
                  }
                }}
                value={parameters.repeatPenalty.toString()}
              >
                <SelectTrigger
                  aria-label="Repeat penalty"
                  className="w-full"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Neutral / Off — 1.0</SelectItem>
                  <SelectItem value="1.1">
                    Mild Discouragement — 1.1
                  </SelectItem>
                  <SelectItem value="1.2">
                    Strong Discouragement — 1.2
                  </SelectItem>
                </SelectContent>
              </Select>
            </section>
          </div>

          <section className="flex items-center justify-between gap-4 rounded-lg border bg-muted/15 p-4">
            <div>
              <h3 className="font-medium">Clone voice</h3>
              <p className="text-muted-foreground">
                Use the selected persona&apos;s audio sample when one is available.
              </p>
            </div>
            <Switch
              aria-label="Clone persona voice"
              checked={parameters.cloneVoice}
              onCheckedChange={(checked) =>
                setParameter('cloneVoice', checked)
              }
            />
          </section>

          <section className="grid gap-3 rounded-lg border bg-muted/15 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="font-medium">TTS generation steps</h3>
                <p className="text-muted-foreground">
                  Controls OmniVoice&apos;s denoising effort. Higher values can
                  improve speech quality, but take longer to generate.
                </p>
              </div>
              <output className="min-w-20 rounded-md border bg-background px-2 py-1 text-center font-mono tabular-nums">
                {parameters.ttsStepLevel} / 10
                <span className="block text-[10px] text-muted-foreground">
                  {omnivoiceNumStepsFromLevel(parameters.ttsStepLevel)} steps
                </span>
              </output>
            </div>
            <Slider
              aria-label="TTS generation steps"
              max={10}
              min={1}
              onValueChange={([value]) =>
                setParameter('ttsStepLevel', Math.round(value))
              }
              step={1}
              value={[parameters.ttsStepLevel]}
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>Faster · 1</span>
              <span>Higher quality · 10</span>
            </div>
          </section>

          <section className="grid gap-3 rounded-lg border bg-muted/15 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="font-medium">Maximum new tokens</h3>
                <p className="text-muted-foreground">
                  Sets the response-length ceiling on a logarithmic scale.
                </p>
              </div>
              <output className="min-w-16 rounded-md border bg-background px-2 py-1 text-center font-mono tabular-nums">
                {parameters.maxNewTokens.toLocaleString()}
              </output>
            </div>

            <div className="relative pt-2">
              <Slider
                aria-label="Maximum new tokens"
                className="[&_[data-slot=slider-thumb]]:relative [&_[data-slot=slider-thumb]]:z-20"
                max={maxTokenExponent}
                min={minTokenExponent}
                onValueChange={([exponent]) =>
                  setParameter('maxNewTokens', exponentToTokens(exponent))
                }
                step={0.01}
                value={[tokensToExponent(parameters.maxNewTokens)]}
              />
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-[7px] top-[11px] z-10 h-0"
              >
                {tokenCheckpoints.map((checkpoint) => (
                  <span
                    className="absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-primary bg-background shadow-sm"
                    key={checkpoint}
                    style={{ left: checkpointPosition(checkpoint) }}
                  />
                ))}
              </div>
              <div className="relative mt-3 h-4 text-[9px] text-muted-foreground">
                {tokenCheckpoints.map((checkpoint) => (
                  <span
                    className="absolute -translate-x-1/2 tabular-nums"
                    key={checkpoint}
                    style={{ left: checkpointPosition(checkpoint) }}
                  >
                    {checkpoint >= 1024 ? `${checkpoint / 1024}k` : checkpoint}
                  </span>
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>64</span>
                <span>8,192</span>
              </div>
            </div>
          </section>
        </div>

        <div className="flex items-center justify-between gap-3">
          <p className="text-[10px] text-muted-foreground">
            Changes are saved automatically.
          </p>
          <Button
            disabled={
              parameters.temperature ===
                defaultGenerationParameters.temperature &&
              parameters.cloneVoice === defaultGenerationParameters.cloneVoice &&
              parameters.maxNewTokens ===
                defaultGenerationParameters.maxNewTokens &&
              parameters.ttsStepLevel ===
                defaultGenerationParameters.ttsStepLevel &&
              parameters.repeatPenalty ===
                defaultGenerationParameters.repeatPenalty &&
              parameters.seed === defaultGenerationParameters.seed
            }
            onClick={resetParameters}
            size="sm"
            variant="outline"
          >
            Reset defaults
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
