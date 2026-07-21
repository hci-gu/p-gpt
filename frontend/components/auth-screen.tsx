import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAuthStore } from '@/src/state/auth'
import { KeyRoundIcon, MailCheckIcon, MessageSquareIcon } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'

type AuthMode = 'sign-in' | 'sign-up' | 'reset'

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Something went wrong. Try again.'

export function AuthScreen() {
  const requestPasswordReset = useAuthStore(
    (state) => state.requestPasswordReset
  )
  const requestVerification = useAuthStore(
    (state) => state.requestVerification
  )
  const signIn = useAuthStore((state) => state.signIn)
  const signUp = useAuthStore((state) => state.signUp)
  const [mode, setMode] = useState<AuthMode>('sign-in')
  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const beginRequest = () => {
    setError(null)
    setNotice(null)
    setIsSubmitting(true)
  }

  const handleSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    const submittedEmail = String(data.get('email') ?? '')
    beginRequest()

    try {
      await signIn(submittedEmail, String(data.get('password') ?? ''))
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSignUp = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    const submittedEmail = String(data.get('email') ?? '')
    beginRequest()

    try {
      await signUp(
        submittedEmail,
        String(data.get('password') ?? ''),
        String(data.get('passwordConfirm') ?? '')
      )
      setEmail(submittedEmail)
      setMode('sign-in')

      try {
        await requestVerification(submittedEmail)
        setNotice('Account created. Check your email before signing in.')
      } catch {
        setNotice(
          'Account created, but the verification email could not be sent. Use “Resend verification” below.'
        )
      }
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handlePasswordReset = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    const submittedEmail = String(data.get('email') ?? '')
    beginRequest()

    try {
      await requestPasswordReset(submittedEmail)
      setEmail(submittedEmail)
      setNotice(
        'If an account exists for that email, a password reset link has been sent.'
      )
      setMode('sign-in')
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleResendVerification = async () => {
    if (!email.trim()) {
      setError('Enter your email address first.')
      return
    }

    beginRequest()
    try {
      await requestVerification(email)
      setNotice('If the account is unverified, a new email has been sent.')
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-muted/40 px-4 py-10">
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 flex size-11 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            <MessageSquareIcon className="size-5" />
          </div>
          <CardTitle className="text-xl">Welcome to P-GPT</CardTitle>
          <CardDescription>
            Sign in to keep your chats and personas private.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {notice && (
            <Alert className="mb-4">
              <MailCheckIcon />
              <AlertTitle>Check your email</AlertTitle>
              <AlertDescription>{notice}</AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert className="mb-4" variant="destructive">
              <AlertTitle>Could not complete the request</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {mode === 'reset' ? (
            <form className="grid gap-4" onSubmit={handlePasswordReset}>
              <div className="text-center">
                <KeyRoundIcon className="mx-auto mb-2 size-5" />
                <h2 className="font-medium">Reset your password</h2>
                <p className="text-muted-foreground">
                  PocketBase will email you a secure reset link.
                </p>
              </div>
              <label className="grid gap-1.5">
                <span className="font-medium">Email</span>
                <Input
                  autoComplete="email"
                  name="email"
                  onChange={(event) => setEmail(event.target.value)}
                  required
                  type="email"
                  value={email}
                />
              </label>
              <Button disabled={isSubmitting} size="lg" type="submit">
                {isSubmitting && <Spinner />}
                Send reset link
              </Button>
              <Button
                onClick={() => {
                  setError(null)
                  setMode('sign-in')
                }}
                type="button"
                variant="ghost"
              >
                Back to sign in
              </Button>
            </form>
          ) : (
            <Tabs
              onValueChange={(value) => {
                setError(null)
                setNotice(null)
                setMode(value as AuthMode)
              }}
              value={mode}
            >
              <TabsList className="mb-4 grid w-full grid-cols-2">
                <TabsTrigger value="sign-in">Sign in</TabsTrigger>
                <TabsTrigger value="sign-up">Create account</TabsTrigger>
              </TabsList>

              <TabsContent value="sign-in">
                <form className="grid gap-4" onSubmit={handleSignIn}>
                  <label className="grid gap-1.5">
                    <span className="font-medium">Email</span>
                    <Input
                      autoComplete="email"
                      name="email"
                      onChange={(event) => setEmail(event.target.value)}
                      required
                      type="email"
                      value={email}
                    />
                  </label>
                  <label className="grid gap-1.5">
                    <span className="font-medium">Password</span>
                    <Input
                      autoComplete="current-password"
                      minLength={8}
                      name="password"
                      required
                      type="password"
                    />
                  </label>
                  <Button disabled={isSubmitting} size="lg" type="submit">
                    {isSubmitting && <Spinner />}
                    Sign in
                  </Button>
                  <div className="flex flex-wrap justify-center gap-x-1 text-muted-foreground">
                    <Button
                      onClick={() => {
                        setError(null)
                        setNotice(null)
                        setMode('reset')
                      }}
                      type="button"
                      variant="link"
                    >
                      Forgot password?
                    </Button>
                    <Button
                      disabled={isSubmitting}
                      onClick={() => void handleResendVerification()}
                      type="button"
                      variant="link"
                    >
                      Resend verification
                    </Button>
                  </div>
                </form>
              </TabsContent>

              <TabsContent value="sign-up">
                <form className="grid gap-4" onSubmit={handleSignUp}>
                  <label className="grid gap-1.5">
                    <span className="font-medium">Email</span>
                    <Input
                      autoComplete="email"
                      name="email"
                      onChange={(event) => setEmail(event.target.value)}
                      required
                      type="email"
                      value={email}
                    />
                  </label>
                  <label className="grid gap-1.5">
                    <span className="font-medium">Password</span>
                    <Input
                      autoComplete="new-password"
                      minLength={8}
                      name="password"
                      required
                      type="password"
                    />
                  </label>
                  <label className="grid gap-1.5">
                    <span className="font-medium">Confirm password</span>
                    <Input
                      autoComplete="new-password"
                      minLength={8}
                      name="passwordConfirm"
                      required
                      type="password"
                    />
                  </label>
                  <Button disabled={isSubmitting} size="lg" type="submit">
                    {isSubmitting && <Spinner />}
                    Create account
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </main>
  )
}
