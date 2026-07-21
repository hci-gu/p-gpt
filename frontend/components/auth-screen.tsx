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
import { MessageSquareIcon } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'

type AuthMode = 'sign-in' | 'sign-up'

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Something went wrong. Try again.'

export function AuthScreen() {
  const signIn = useAuthStore((state) => state.signIn)
  const signUp = useAuthStore((state) => state.signUp)
  const [mode, setMode] = useState<AuthMode>('sign-in')
  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    setError(null)
    setIsSubmitting(true)

    try {
      await signIn(
        String(data.get('email') ?? ''),
        String(data.get('password') ?? '')
      )
    } catch (requestError) {
      setError(getErrorMessage(requestError))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSignUp = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    setError(null)
    setIsSubmitting(true)

    try {
      await signUp(
        String(data.get('email') ?? ''),
        String(data.get('password') ?? ''),
        String(data.get('passwordConfirm') ?? '')
      )
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
          {error && (
            <Alert className="mb-4" variant="destructive">
              <AlertTitle>Could not complete the request</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Tabs
            onValueChange={(value) => {
              setError(null)
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
        </CardContent>
      </Card>
    </main>
  )
}
