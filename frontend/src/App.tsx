import { Button } from '@/components/ui/button'
import { AppSidebar } from '@/components/app-sidebar'
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'

function App() {
  return (
    <>
      <SidebarProvider>
        <AppSidebar />
        <main>
          <SidebarTrigger />
          <div className="flex flex-col items-center justify-center gap-4">
            <Button>Hello world</Button>
          </div>
        </main>
      </SidebarProvider>
    </>
  )
}

export default App
