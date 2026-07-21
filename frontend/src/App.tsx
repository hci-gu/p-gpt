import { AppSidebar } from '@/components/app-sidebar'
import { AuthScreen } from '@/components/auth-screen'
import { SidebarProvider } from '@/components/ui/sidebar'
import { Spinner } from '@/components/ui/spinner'
import { backgroundOptions } from '@/lib/backgrounds'
import { useAuthStore } from '@/src/state/auth'
import { usePreferencesStore } from '@/src/state/preferences'
import { useEffect } from 'react'
import type { CSSProperties } from 'react'
import ChatPage from './pages/Chat'

const backgroundTuning = {
  chatSurfaceOpacity: 0.66,
  imageOpacity: 0.86,
  overlayOpacity: 0.28,
  promptInputSurfaceOpacity: 0.88,
}

const backgroundStyle = {
  '--chat-surface-opacity': backgroundTuning.chatSurfaceOpacity,
  '--image-opacity': backgroundTuning.imageOpacity,
  '--overlay-opacity': backgroundTuning.overlayOpacity,
  '--prompt-input-surface-opacity': backgroundTuning.promptInputSurfaceOpacity,
} as CSSProperties

function App() {
  const authStatus = useAuthStore((state) => state.status)
  const initializeAuth = useAuthStore((state) => state.initialize)
  const user = useAuthStore((state) => state.user)
  const selectedBackgroundId = usePreferencesStore(
    (state) => state.selectedBackgroundId
  )
  const selectedBackground = backgroundOptions.find(
    (option) => option.id === selectedBackgroundId
  )
  const backgroundImageUrl = selectedBackground?.url ?? null

  useEffect(() => {
    void initializeAuth()
  }, [initializeAuth])

  if (authStatus === 'loading') {
    return (
      <main className="flex min-h-screen items-center justify-center bg-background">
        <Spinner className="size-6" />
      </main>
    )
  }

  if (authStatus === 'anonymous') {
    return <AuthScreen />
  }

  return (
    <SidebarProvider key={user?.id}>
      <div
        className="relative isolate flex min-h-screen w-full min-w-0 flex-1 flex-col overflow-hidden bg-background text-foreground"
        style={{
          ...backgroundStyle,
          backgroundColor: backgroundImageUrl ? undefined : '#fff',
        }}
      >
        {backgroundImageUrl && (
          <>
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 -z-20 bg-cover bg-center bg-no-repeat"
              style={{
                backgroundImage: `url(${backgroundImageUrl})`,
                opacity: 'var(--image-opacity)',
              }}
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 -z-10 bg-background"
              style={{ opacity: 'var(--overlay-opacity)' }}
            />
          </>
        )}
        <div className="flex min-h-0 flex-1">
          <AppSidebar />
          <main className="mx-auto flex h-screen w-full max-w-6xl flex-col px-4 py-6 sm:px-6 lg:px-8">
            <section
              className="chat-shell min-h-0 flex-1 overflow-hidden rounded-lg border shadow-sm backdrop-blur-sm"
              style={{
                backgroundColor:
                  'hsl(0 0% 100% / var(--chat-surface-opacity))',
              }}
            >
              <ChatPage />
            </section>
          </main>
        </div>
      </div>
    </SidebarProvider>
  )
}

export default App
