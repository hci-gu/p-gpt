import type { ChatEvaluation, EvaluationMetric } from '@/lib/chat-history'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import { AlertCircleIcon, CheckCircle2Icon, ClipboardCheckIcon } from 'lucide-react'

type EvaluationPhase = 'progress' | 'results' | 'error'

const metricLabels: Array<[keyof Pick<
  ChatEvaluation,
  'practitioner_empathy' | 'practitioner_professionalism' | 'practitioner_relevance'
>, string]> = [
  ['practitioner_empathy', 'Empathy and rapport'],
  ['practitioner_professionalism', 'Professionalism and boundaries'],
  ['practitioner_relevance', 'Relevance and context'],
]

const MetricCard = ({ label, metric }: { label: string; metric: EvaluationMetric }) => (
  <section className="rounded-lg border bg-muted/30 p-3">
    <div className="flex items-center justify-between gap-3">
      <h3 className="font-medium text-sm">{label}</h3>
      <span className="rounded-full bg-primary px-2 py-0.5 font-semibold text-primary-foreground text-xs">
        {metric.score}/5
      </span>
    </div>
    <p className="mt-2 text-muted-foreground text-sm">{metric.rationale}</p>
  </section>
)

export const EvaluationResults = ({ evaluation }: { evaluation: ChatEvaluation }) => (
  <div className="grid gap-3">
    {metricLabels.map(([key, label]) => (
      <MetricCard key={key} label={label} metric={evaluation[key]} />
    ))}
    <section className="rounded-lg border bg-muted/30 p-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-medium text-sm">Mock patient frustration</h3>
        <span className="rounded-full bg-muted px-2 py-0.5 font-semibold text-xs capitalize">
          {evaluation.mock_patient_frustration.level}
        </span>
      </div>
      <p className="mt-2 text-muted-foreground text-sm">
        {evaluation.mock_patient_frustration.rationale}
      </p>
    </section>
    <section className="rounded-lg border bg-primary/5 p-3">
      <h3 className="font-medium text-sm">Learning feedback</h3>
      <p className="mt-2 whitespace-pre-wrap text-muted-foreground text-sm">
        {evaluation.overall_feedback.summary}
      </p>
    </section>
  </div>
)

export const EvaluationDialog = ({
  cloudEvaluation,
  error,
  evaluation,
  message,
  onOpenChange,
  open,
  phase,
  progress,
}: {
  cloudEvaluation: boolean
  error: string | null
  evaluation: ChatEvaluation | null
  message: string
  onOpenChange: (open: boolean) => void
  open: boolean
  phase: EvaluationPhase
  progress: number
}) => (
  <Dialog onOpenChange={onOpenChange} open={open}>
    <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-xl">
      {phase === 'progress' && (
        <>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ClipboardCheckIcon className="size-5" /> Evaluating conversation
            </DialogTitle>
            <DialogDescription>
              Reviewing the practitioner’s communication with the simulated patient.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3 py-3">
            {cloudEvaluation && (
              <p className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-amber-950 text-sm dark:text-amber-100">
                This evaluation will be processed by the configured cloud evaluator.
              </p>
            )}
            <Progress value={progress} />
            <p className="text-muted-foreground text-sm" role="status">
              {message}
            </p>
            <p className="text-muted-foreground text-xs">{progress}% complete</p>
          </div>
        </>
      )}
      {phase === 'results' && evaluation && (
        <>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle2Icon className="size-5 text-green-600" /> Evaluation complete
            </DialogTitle>
            <DialogDescription>
              This completed training conversation is now read-only.
            </DialogDescription>
          </DialogHeader>
          <EvaluationResults evaluation={evaluation} />
        </>
      )}
      {phase === 'error' && (
        <>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertCircleIcon className="size-5 text-destructive" /> Evaluation unavailable
            </DialogTitle>
            <DialogDescription>{error ?? 'Evaluation failed. Please try again.'}</DialogDescription>
          </DialogHeader>
          <Button onClick={() => onOpenChange(false)} type="button">Close</Button>
        </>
      )}
    </DialogContent>
  </Dialog>
)
