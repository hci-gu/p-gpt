import type { ChatHistoryRecord } from '@/lib/chat-history'
import {
  CHAT_HISTORY_UPDATED_EVENT,
  deleteChatHistory,
  listChatHistories,
  renameChatHistory,
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
  PencilIcon,
  PlusIcon,
  Settings2Icon,
  Trash2Icon,
  UsersIcon,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
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
  const [editingRecordId, setEditingRecordId] = useState<string | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  const [renamingRecordId, setRenamingRecordId] = useState<string | null>(null)
  const [renameErrorId, setRenameErrorId] = useState<string | null>(null)
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isBackgroundDialogOpen, setIsBackgroundDialogOpen] = useState(false)
  const [isPersonaDialogOpen, setIsPersonaDialogOpen] = useState(false)
  const [isParametersDialogOpen, setIsParametersDialogOpen] = useState(false)

  const loadChatHistories = useCallback(async (signal?: AbortSignal) => {
    try {
      const records = await listChatHistories(signal)
      const latestRecords = records.sort(
        (first, second) =>
          new Date(second.updated).getTime() -
          new Date(first.updated).getTime()
      )

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
        setEditingRecordId((current) =>
          current === recordId ? null : current
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

  const handleStartRename = useCallback((record: ChatHistoryRecord) => {
    setEditingRecordId(record.id)
    setRenameDraft(record.title)
    setRenameErrorId(null)
  }, [])

  const handleRenameChat = useCallback(
    async (record: ChatHistoryRecord) => {
      const title = renameDraft.trim()
      if (!title) {
        setRenameErrorId(record.id)
        return
      }
      if (title === record.title) {
        setEditingRecordId(null)
        setRenameErrorId(null)
        return
      }

      setRenamingRecordId(record.id)
      setRenameErrorId(null)
      try {
        await renameChatHistory(record.id, title)
        setChatHistories((records) =>
          records.map((current) =>
            current.id === record.id ? { ...current, title } : current
          )
        )
        setEditingRecordId(null)
      } catch {
        setRenameErrorId(record.id)
      } finally {
        setRenamingRecordId(null)
      }
    },
    [renameDraft]
  )

  const visibleChatHistories = useMemo(
    () =>
      isHistoryExpanded ? chatHistories : chatHistories.slice(0, 5),
    [chatHistories, isHistoryExpanded]
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
            <div
              className={
                isHistoryExpanded && chatHistories.length > 5
                  ? 'max-h-[50vh] overflow-y-auto overscroll-contain pr-1'
                  : undefined
              }
            >
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
                visibleChatHistories.map((record) => {
                  const isEditing = editingRecordId === record.id
                  return (
                    <SidebarMenuItem key={record.id}>
                      {isEditing ? (
                        <form
                          className={`flex min-h-10 w-full items-center gap-2 rounded-[calc(var(--radius-sm)+2px)] p-2 pr-14 text-xs ${
                            activeHistoryId === record.id
                              ? 'bg-sidebar-accent'
                              : ''
                          }`}
                          onSubmit={(event) => {
                            event.preventDefault()
                            void handleRenameChat(record)
                          }}
                        >
                          <MessageSquareIcon className="size-4 shrink-0" />
                          <input
                            aria-label={`Rename ${record.title}`}
                            autoFocus
                            className={`h-7 min-w-0 flex-1 rounded border bg-background px-2 text-xs outline-none focus:ring-2 focus:ring-sidebar-ring ${
                              renameErrorId === record.id
                                ? 'border-destructive'
                                : 'border-sidebar-border'
                            }`}
                            disabled={renamingRecordId === record.id}
                            maxLength={120}
                            onBlur={() => {
                              if (renamingRecordId !== record.id) {
                                setEditingRecordId(null)
                                setRenameErrorId(null)
                              }
                            }}
                            onChange={(event) => setRenameDraft(event.target.value)}
                            onFocus={(event) => event.currentTarget.select()}
                            onKeyDown={(event) => {
                              if (event.key === 'Escape') {
                                setEditingRecordId(null)
                                setRenameErrorId(null)
                              }
                            }}
                            title={
                              renameErrorId === record.id
                                ? 'Could not rename chat. Use a non-empty title and try again.'
                                : 'Press Enter to save'
                            }
                            value={renameDraft}
                          />
                        </form>
                      ) : (
                        <SidebarMenuButton
                          aria-current={
                            activeHistoryId === record.id ? 'page' : undefined
                          }
                          className="h-auto min-h-10 items-start pr-14"
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
                      )}
                      <div
                        className={`absolute inset-y-0 right-1 flex items-center gap-0.5 transition-opacity group-data-[collapsible=icon]:hidden ${
                          isEditing
                            ? 'opacity-100'
                            : 'group-focus-within/menu-item:opacity-100 group-hover/menu-item:opacity-100 md:opacity-0'
                        }`}
                      >
                        <button
                          aria-label={`Rename ${record.title}`}
                          className={`flex size-5 items-center justify-center rounded-[calc(var(--radius-sm)-2px)] outline-none hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring [&>svg]:size-4 ${
                            renameErrorId === record.id
                              ? 'text-destructive'
                              : 'text-sidebar-foreground'
                          }`}
                          disabled={renamingRecordId === record.id}
                          onClick={(event) => {
                            event.stopPropagation()
                            handleStartRename(record)
                          }}
                          title="Rename chat"
                          type="button"
                        >
                          <PencilIcon />
                        </button>
                        <button
                          aria-label={`Delete ${record.title}`}
                          className={`flex size-5 items-center justify-center rounded-[calc(var(--radius-sm)-2px)] outline-none hover:bg-destructive/15 hover:text-destructive focus-visible:ring-2 focus-visible:ring-sidebar-ring focus-visible:text-destructive [&>svg]:size-4 ${
                            deleteErrorId === record.id
                              ? 'bg-destructive/15 text-destructive'
                              : 'text-sidebar-foreground'
                          }`}
                          disabled={deletingRecordId === record.id}
                          onClick={(event) => {
                            event.stopPropagation()
                            void handleDeleteChat(record.id)
                          }}
                          title={
                            deleteErrorId === record.id
                              ? 'Could not delete chat. Try again.'
                              : 'Delete chat'
                          }
                          type="button"
                        >
                          <Trash2Icon />
                        </button>
                      </div>
                    </SidebarMenuItem>
                  )
                })}
              </SidebarMenu>
            </div>
            {!isLoading && !loadError && chatHistories.length > 5 && (
              <button
                className="mt-1 px-2 text-left text-xs text-sidebar-foreground/65 hover:text-sidebar-foreground hover:underline group-data-[collapsible=icon]:hidden"
                onClick={() => setIsHistoryExpanded((expanded) => !expanded)}
                type="button"
              >
                {isHistoryExpanded ? 'Show less' : 'Load more'}
              </button>
            )}
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
                    {background.thumbnailUrl ? (
                      <img
                        alt=""
                        className="size-full object-cover transition-transform duration-200 group-hover/background:scale-105"
                        src={background.thumbnailUrl}
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
