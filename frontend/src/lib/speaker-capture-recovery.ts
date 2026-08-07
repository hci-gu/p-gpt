export const speakerCaptureStallMilliseconds = 5_000
export const speakerMaximumPendingVadFrames = 64
export const speakerRecoveryLimit = 3
export const speakerRecoveryWindowMilliseconds = 60_000

export class SpeakerRecoveryWindow {
  private failures: number[] = []

  record(now: number) {
    this.failures = this.failures.filter(
      (failure) => now - failure < speakerRecoveryWindowMilliseconds
    )
    this.failures.push(now)
    return {
      count: this.failures.length,
      exhausted: this.failures.length >= speakerRecoveryLimit,
    }
  }

  reset() {
    this.failures = []
  }
}
