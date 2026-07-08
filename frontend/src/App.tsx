import { AppSidebar } from '@/components/app-sidebar'
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import type { CSSProperties } from 'react'
import backgroundImageUrl from '../assets/background.png'
import ChatPage from './pages/Chat'

const backgroundTuning = {
  chatSurfaceOpacity: 0.66,
  headerSurfaceOpacity: 0.72,
  imageOpacity: 0.86,
  overlayOpacity: 0.28,
  promptInputSurfaceOpacity: 0.88,
}

const backgroundStyle = {
  '--chat-surface-opacity': backgroundTuning.chatSurfaceOpacity,
  '--header-surface-opacity': backgroundTuning.headerSurfaceOpacity,
  '--image-opacity': backgroundTuning.imageOpacity,
  '--overlay-opacity': backgroundTuning.overlayOpacity,
  '--prompt-input-surface-opacity': backgroundTuning.promptInputSurfaceOpacity,
} as CSSProperties

function App() {
  return (
    <SidebarProvider>
      <div
        className="relative isolate flex min-h-screen w-full min-w-0 flex-1 flex-col overflow-hidden bg-background text-foreground"
        style={backgroundStyle}
      >
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
        <header
          className="relative z-20 border-b backdrop-blur-md"
          style={{
            backgroundColor: 'hsl(0 0% 100% / var(--header-surface-opacity))',
          }}
        >
          <nav className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">
            <div className="flex min-w-0 items-center gap-3">
              <SidebarTrigger className="-ml-1" />
              <a className="truncate font-semibold tracking-tight" href="/">
                P-GPT
              </a>
            </div>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <a className="transition-colors hover:text-foreground" href="/">
                Chat
              </a>
              <a className="transition-colors hover:text-foreground" href="/">
                Docs
              </a>
            </div>
          </nav>
        </header>

        <div className="flex min-h-0 flex-1">
          <AppSidebar className="!top-14 !bottom-auto !h-[calc(100svh-3.5rem)]" />
          <main className="mx-auto flex h-[calc(100vh-3.5rem)] w-full max-w-6xl flex-col px-4 py-6 sm:px-6 lg:px-8">
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
