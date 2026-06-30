import { AppSidebar } from '@/components/app-sidebar'
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import ChatPage from './pages/Chat'

function App() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen min-w-0 flex-1 flex-col bg-background text-foreground">
        <header className="relative z-20 border-b bg-background/95">
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
            <section className="min-h-0 flex-1 overflow-hidden rounded-lg border bg-card shadow-sm">
              <ChatPage />
            </section>
          </main>
        </div>
      </div>
    </SidebarProvider>
  )
}

export default App
