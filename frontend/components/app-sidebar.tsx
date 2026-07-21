import type { ChatHistoryRecord } from '@/lib/chat-history'
import {
  CHAT_HISTORY_UPDATED_EVENT,
  deleteChatHistory,
  listChatHistories,
} from '@/lib/chat-history'
import { useChatStore } from '@/src/state/chat'
import { useAuthStore } from '@/src/state/auth'
import { Button } from '@/components/ui/button'
import { ParametersDialog } from '@/components/parameters-dialog'
import { PersonaDialog } from '@/components/persona-dialog'
import {
  Collapsible,
  CollapsibleContent,
} from '@/components/ui/collapsible'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  useSidebar,
} from '@/components/ui/sidebar'
import { backgroundOptions } from '@/lib/backgrounds'
import { usePreferencesStore } from '@/src/state/preferences'
import {
  CheckIcon,
  BrainIcon,
  ChevronUpIcon,
  ImageIcon,
  LogOutIcon,
  MessageSquareIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  PlusIcon,
  Settings2Icon,
  Trash2Icon,
  UsersIcon,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import type { ComponentProps } from 'react'

const getChatTimestamp = (record: ChatHistoryRecord) => {
  const timestamp = record.updated || record.created
  if (!timestamp) {
    return ''
  }

  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return new Intl.DateTimeFormat(undefined, {
    day: 'numeric',
    month: 'short',
  }).format(date)
}

export function AppSidebar(props: ComponentProps<typeof Sidebar>) {
  const { open, setOpen } = useSidebar()
  const activeHistoryId = useChatStore((state) => state.activeHistoryId)
  const loadChat = useChatStore((state) => state.loadChat)
  const startNewChat = useChatStore((state) => state.startNewChat)
  const clearDeletedChat = useChatStore((state) => state.clearDeletedChat)
  const signOut = useAuthStore((state) => state.signOut)
  const user = useAuthStore((state) => state.user)
  const selectedBackgroundId = usePreferencesStore(
    (state) => state.selectedBackgroundId
  )
  const selectBackground = usePreferencesStore(
    (state) => state.selectBackground
  )
  const [chatHistories, setChatHistories] = useState<ChatHistoryRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState(false)
  const [deletingRecordId, setDeletingRecordId] = useState<string | null>(null)
  const [deleteErrorId, setDeleteErrorId] = useState<string | null>(null)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isBackgroundDialogOpen, setIsBackgroundDialogOpen] = useState(false)
  const [isPersonaDialogOpen, setIsPersonaDialogOpen] = useState(false)
  const [isParametersDialogOpen, setIsParametersDialogOpen] = useState(false)

  const loadChatHistories = useCallback(async (signal?: AbortSignal) => {
    try {
      const records = await listChatHistories(signal)
      const latestRecords = records
        .sort(
          (first, second) =>
            new Date(second.updated).getTime() -
            new Date(first.updated).getTime()
        )
        .slice(0, 5)

      setChatHistories(latestRecords)
      setLoadError(false)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      setLoadError(true)
    } finally {
      if (!signal?.aborted) {
        setIsLoading(false)
      }
    }
  }, [])

  const handleDeleteChat = useCallback(
    async (recordId: string) => {
      setDeletingRecordId(recordId)
      setDeleteErrorId(null)

      try {
        await deleteChatHistory(recordId)
        setChatHistories((records) =>
          records.filter((record) => record.id !== recordId)
        )

        clearDeletedChat(recordId)
      } catch {
        setDeleteErrorId(recordId)
      } finally {
        setDeletingRecordId(null)
      }
    },
    [clearDeletedChat]
  )

  useEffect(() => {
    const abortController = new AbortController()
    void loadChatHistories(abortController.signal)

    const handleHistoryUpdated = () => {
      void loadChatHistories()
    }
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, handleHistoryUpdated)

    return () => {
      abortController.abort()
      window.removeEventListener(
        CHAT_HISTORY_UPDATED_EVENT,
        handleHistoryUpdated
      )
    }
  }, [loadChatHistories])

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader className="border-b border-sidebar-border">
        <div className="flex h-8 items-center gap-2 overflow-hidden">
          <Button
            aria-label={open ? 'Collapse sidebar' : 'Expand sidebar'}
            className="size-8 shrink-0"
            onClick={() => setOpen(!open)}
            size="icon-sm"
            title={open ? 'Collapse sidebar' : 'Expand sidebar'}
            variant="ghost"
          >
            {open ? <PanelLeftCloseIcon /> : <PanelLeftOpenIcon />}
          </Button>
          <span className="truncate font-semibold tracking-tight group-data-[collapsible=icon]:hidden">
            P-GPT
          </span>
        </div>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              className="bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90 hover:text-sidebar-primary-foreground"
              onClick={startNewChat}
              tooltip="New chat"
            >
              <PlusIcon />
              <span>New chat</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Recent chats</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {isLoading &&
                Array.from({ length: 3 }, (_, index) => (
                  <SidebarMenuItem key={index}>
                    <SidebarMenuSkeleton showIcon />
                  </SidebarMenuItem>
                ))}
              {!isLoading && loadError && (
                <SidebarMenuItem>
                  <SidebarMenuButton disabled tooltip="Could not load chats">
                    <MessageSquareIcon />
                    <span>Could not load chats</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              )}
              {!isLoading && !loadError && chatHistories.length === 0 && (
                <SidebarMenuItem>
                  <SidebarMenuButton disabled tooltip="No recent chats">
                    <MessageSquareIcon />
                    <span>No recent chats</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              )}
              {!isLoading &&
                !loadError &&
                chatHistories.map((record) => {
                  return (
                    <SidebarMenuItem key={record.id}>
                      <SidebarMenuButton
                        aria-current={
                          activeHistoryId === record.id ? 'page' : undefined
                        }
                        className="h-auto min-h-10 items-start"
                        isActive={activeHistoryId === record.id}
                        onClick={() =>
                          loadChat(
                            record.id,
                            record.conversation,
                            record.personaId
                          )
                        }
                        tooltip={record.title}
                      >
                        <MessageSquareIcon className="mt-0.5" />
                        <span className="grid min-w-0 flex-1 leading-tight">
                          <span className="truncate">{record.title}</span>
                          <span className="mt-0.5 text-[10px] text-sidebar-foreground/55">
                            {getChatTimestamp(record)}
                          </span>
                        </span>
                      </SidebarMenuButton>
                      <SidebarMenuAction
                        aria-label={`Delete ${record.title}`}
                        className={`top-2.5 hover:bg-destructive/15 hover:text-destructive focus-visible:text-destructive ${
                          deleteErrorId === record.id
                            ? 'bg-destructive/15 text-destructive'
                            : ''
                        }`}
                        disabled={deletingRecordId === record.id}
                        onClick={(event) => {
                          event.stopPropagation()
                          void handleDeleteChat(record.id)
                        }}
                        showOnHover
                        title={
                          deleteErrorId === record.id
                            ? 'Could not delete chat. Try again.'
                            : 'Delete chat'
                        }
                      >
                        <Trash2Icon />
                      </SidebarMenuAction>
                    </SidebarMenuItem>
                  )
                })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="border-t border-sidebar-border">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={signOut} tooltip="Sign out">
              <LogOutIcon />
              <span className="truncate">{user?.email || 'Sign out'}</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <Collapsible onOpenChange={setIsSettingsOpen} open={isSettingsOpen}>
          <CollapsibleContent>
            <SidebarMenu className="pb-1">
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => setIsPersonaDialogOpen(true)}
                  tooltip="Choose persona"
                >
                  <UsersIcon />
                  <span>Personas</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => setIsParametersDialogOpen(true)}
                  tooltip="Tune parameters"
                >
                  <BrainIcon />
                  <span>Parameters</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => setIsBackgroundDialogOpen(true)}
                  tooltip="Change background"
                >
                  <ImageIcon />
                  <span>Background</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </CollapsibleContent>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                aria-expanded={isSettingsOpen}
                onClick={() => {
                  if (!open) {
                    setOpen(true)
                  }
                  setIsSettingsOpen((isOpen) => !isOpen)
                }}
                tooltip="Settings"
              >
                <Settings2Icon />
                <span>Settings</span>
                <ChevronUpIcon
                  className={`ml-auto transition-transform group-data-[collapsible=icon]:hidden ${
                    isSettingsOpen ? 'rotate-180' : ''
                  }`}
                />
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </Collapsible>
      </SidebarFooter>

      <Dialog
        onOpenChange={setIsBackgroundDialogOpen}
        open={isBackgroundDialogOpen}
      >
        <DialogContent className="max-h-[85vh] overflow-hidden sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>Choose a background</DialogTitle>
            <DialogDescription>
              Select a background for the chat workspace.
            </DialogDescription>
          </DialogHeader>
          <div className="grid max-h-[65vh] grid-cols-2 gap-3 overflow-y-auto pr-1 sm:grid-cols-3">
            {backgroundOptions.map((background) => {
              const isSelected = selectedBackgroundId === background.id

              return (
                <button
                  aria-label={`Use ${background.label.toLowerCase()}`}
                  aria-pressed={isSelected}
                  className="group/background relative overflow-hidden rounded-lg border bg-muted text-left shadow-sm outline-none transition hover:border-foreground/40 hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring data-[selected=true]:border-foreground"
                  data-selected={isSelected}
                  key={background.id}
                  onClick={() => {
                    selectBackground(background.id)
                    setIsBackgroundDialogOpen(false)
                  }}
                  type="button"
                >
                  <div className="aspect-[16/10] overflow-hidden bg-white">
                    {background.url ? (
                      <img
                        alt=""
                        className="size-full object-cover transition-transform duration-200 group-hover/background:scale-105"
                        src={background.url}
                      />
                    ) : (
                      <div className="size-full bg-white" />
                    )}
                  </div>
                  <div className="flex items-center gap-2 border-t bg-background px-2.5 py-2">
                    <span className="min-w-0 flex-1 truncate text-xs">
                      {background.label}
                    </span>
                    {isSelected && (
                      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <CheckIcon className="size-3" />
                        <span className="sr-only">Selected</span>
                      </span>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        </DialogContent>
      </Dialog>
      <PersonaDialog
        onOpenChange={setIsPersonaDialogOpen}
        open={isPersonaDialogOpen}
      />
      <ParametersDialog
        onOpenChange={setIsParametersDialogOpen}
        open={isParametersDialogOpen}
      />
    </Sidebar>
  )
}
