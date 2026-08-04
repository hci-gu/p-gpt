import defaultProfilePictureUrl from '@/assets/default-person-pfp.png'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import {
  getPersonaPreparation,
  startPersonaPreparation,
  type PersonaRecord,
} from '@/lib/personas'
import { useChatStore } from '@/src/state/chat'
import { usePersonasStore } from '@/src/state/personas'
import {
  CheckIcon,
  LockKeyholeIcon,
  Music2Icon,
  PencilIcon,
  PlusIcon,
  SaveIcon,
  UsersIcon,
} from 'lucide-react'
import type { FormEvent } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'

type PersonaDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const getInitials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('') || '?'

export function PersonaDialog({ open, onOpenChange }: PersonaDialogProps) {
  const personas = usePersonasStore((state) => state.personas)
  const selectedPersonaId = usePersonasStore(
    (state) => state.selectedPersonaId
  )
  const isLoading = usePersonasStore((state) => state.isLoading)
  const loadError = usePersonasStore((state) => state.loadError)
  const ensurePersonasLoaded = usePersonasStore(
    (state) => state.ensurePersonasLoaded
  )
  const createPersona = usePersonasStore((state) => state.createPersona)
  const updatePersona = usePersonasStore((state) => state.updatePersona)
  const selectPersona = usePersonasStore((state) => state.selectPersona)
  const selectPersonaForNewChat = useChatStore(
    (state) => state.selectPersonaForNewChat
  )
  const [activeTab, setActiveTab] = useState('choose')
  const [playingPersonaId, setPlayingPersonaId] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [editingPersona, setEditingPersona] = useState<PersonaRecord | null>(
    null
  )
  const [isSaving, setIsSaving] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  const [preparationId, setPreparationId] = useState<string | null>(null)
  const [preparationMessage, setPreparationMessage] = useState<string | null>(
    null
  )
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    void ensurePersonasLoaded().catch(() => {
      // The dialog renders the store's actionable retry state.
    })
  }, [ensurePersonasLoaded])

  useEffect(
    () => () => {
      audioRef.current?.pause()
      audioRef.current = null
    },
    []
  )

  useEffect(() => {
    if (!preparationId) {
      return undefined
    }

    let isCurrent = true
    let timeoutId: number | null = null

    const poll = async () => {
      try {
        const preparation = await getPersonaPreparation(preparationId)
        if (!isCurrent) {
          return
        }

        if (preparation.status === 'pending') {
          timeoutId = window.setTimeout(poll, 1000)
          return
        }

        setPreparationId(null)
        setPreparationMessage(
          preparation.status === 'ready'
            ? 'Persona changes are ready to use.'
            : `Changes were saved, but preparation failed: ${
                preparation.error ?? 'unknown error'
              }. It will retry when the persona is used.`
        )
      } catch (error) {
        if (!isCurrent) {
          return
        }
        setPreparationId(null)
        setPreparationMessage(
          `Changes were saved, but preparation could not be checked: ${
            error instanceof Error ? error.message : 'it will retry when used.'
          }. It will retry when the persona is used.`
        )
      }
    }

    void poll()

    return () => {
      isCurrent = false
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [preparationId])

  const handleSelectPersona = useCallback(
    (personaId: string) => {
      selectPersona(personaId)
      selectPersonaForNewChat(personaId)
      onOpenChange(false)
    },
    [onOpenChange, selectPersona, selectPersonaForNewChat]
  )

  const handleAudioPreview = useCallback(
    (persona: PersonaRecord) => {
      if (!persona.audioSampleUrl) {
        return
      }

      if (playingPersonaId === persona.id) {
        audioRef.current?.pause()
        audioRef.current = null
        setPlayingPersonaId(null)
        return
      }

      audioRef.current?.pause()
      const audio = new Audio(persona.audioSampleUrl)
      audioRef.current = audio
      setPlayingPersonaId(persona.id)
      audio.addEventListener('ended', () => {
        audioRef.current = null
        setPlayingPersonaId(null)
      })
      void audio.play().catch(() => {
        audioRef.current = null
        setPlayingPersonaId(null)
      })
    },
    [playingPersonaId]
  )

  const handleCreatePersona = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      const form = event.currentTarget
      const data = new FormData(form)
      const profilePicture = data.get('profile_picture')
      const audioSample = data.get('audio_sample')

      setIsCreating(true)
      setCreateError(null)

      try {
        const persona = await createPersona({
          name: String(data.get('name') ?? ''),
          description: String(data.get('description') ?? ''),
          instructionPrompt: String(data.get('instruction_prompt') ?? ''),
          profilePicture:
            profilePicture instanceof File && profilePicture.size > 0
              ? profilePicture
              : null,
          audioSample:
            audioSample instanceof File && audioSample.size > 0
              ? audioSample
              : null,
        })
        selectPersonaForNewChat(persona.id)
        form.reset()
        setActiveTab('choose')
        onOpenChange(false)
      } catch (error) {
        setCreateError(
          error instanceof Error ? error.message : 'Could not create persona.'
        )
      } finally {
        setIsCreating(false)
      }
    },
    [createPersona, onOpenChange, selectPersonaForNewChat]
  )

  const handleEditPersona = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      if (!editingPersona) {
        return
      }

      const originalPersona = editingPersona
      const data = new FormData(event.currentTarget)
      const profilePicture = data.get('profile_picture')
      const audioSample = data.get('audio_sample')
      const name = String(data.get('name') ?? '').trim()
      const description = String(data.get('description') ?? '').trim()
      const instructionPrompt = String(
        data.get('instruction_prompt') ?? ''
      ).trim()
      const replacementAudioSample =
        audioSample instanceof File && audioSample.size > 0
          ? audioSample
          : null
      const prepareSystemPrompt =
        name !== originalPersona.name ||
        instructionPrompt !== originalPersona.instructionPrompt
      const prepareVoiceClonePrompt = Boolean(replacementAudioSample)

      setIsSaving(true)
      setEditError(null)
      setPreparationId(null)
      setPreparationMessage(null)

      try {
        const persona = await updatePersona(originalPersona.id, {
          name,
          description,
          instructionPrompt,
          profilePicture:
            profilePicture instanceof File && profilePicture.size > 0
              ? profilePicture
              : null,
          audioSample: replacementAudioSample,
        })
        setEditingPersona(persona)

        if (prepareSystemPrompt || prepareVoiceClonePrompt) {
          try {
            const preparation = await startPersonaPreparation({
              personaId: persona.id,
              name: persona.name,
              instructionPrompt: persona.instructionPrompt,
              previousAudioSampleUrl: originalPersona.audioSampleUrl,
              audioSampleUrl: persona.audioSampleUrl,
              prepareSystemPrompt,
              prepareVoiceClonePrompt,
            })
            setPreparationId(
              preparation.status === 'pending' ? preparation.id : null
            )
            setPreparationMessage(
              preparation.status === 'pending'
                ? 'Changes saved. Preparing the updated persona…'
                : 'Persona changes are ready to use.'
            )
          } catch (error) {
            setPreparationMessage(
              `Changes were saved, but preparation could not start: ${
                error instanceof Error ? error.message : 'unknown error'
              }. It will retry when the persona is used.`
            )
          }
        } else {
          setPreparationMessage('Changes saved.')
        }
      } catch (error) {
        setEditError(
          error instanceof Error ? error.message : 'Could not save persona.'
        )
      } finally {
        setIsSaving(false)
      }
    },
    [editingPersona, updatePersona]
  )

  const handleStartEditing = useCallback((persona: PersonaRecord) => {
    setEditingPersona(persona)
    setEditError(null)
    setPreparationId(null)
    setPreparationMessage(null)
    setActiveTab('edit')
  }, [])

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-h-[88vh] overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Personas</DialogTitle>
          <DialogDescription>
            Choose who you want to talk to, or create a new persona.
          </DialogDescription>
        </DialogHeader>

        <Tabs onValueChange={setActiveTab} value={activeTab}>
          <TabsList
            className={`grid w-full ${editingPersona ? 'grid-cols-3' : 'grid-cols-2'}`}
          >
            <TabsTrigger value="choose">
              <UsersIcon />
              Choose persona
            </TabsTrigger>
            <TabsTrigger value="create">
              <PlusIcon />
              Create persona
            </TabsTrigger>
            {editingPersona && (
              <TabsTrigger value="edit">
                <PencilIcon />
                Edit persona
              </TabsTrigger>
            )}
          </TabsList>

          <TabsContent className="min-h-0" value="choose">
            {isLoading && (
              <div className="flex min-h-48 items-center justify-center">
                <Spinner />
              </div>
            )}

            {!isLoading && loadError && (
              <div className="flex min-h-48 flex-col items-center justify-center gap-3 text-center">
                <p className="text-muted-foreground">{loadError}</p>
                <Button
                  onClick={() => {
                    void ensurePersonasLoaded().catch(() => undefined)
                  }}
                  size="sm"
                  variant="outline"
                >
                  Try again
                </Button>
              </div>
            )}

            {!isLoading && !loadError && personas.length === 0 && (
              <div className="flex min-h-48 flex-col items-center justify-center gap-3 text-center">
                <p className="text-muted-foreground">No personas yet.</p>
                <Button onClick={() => setActiveTab('create')} size="sm">
                  <PlusIcon />
                  Create your first persona
                </Button>
              </div>
            )}

            {!isLoading && !loadError && personas.length > 0 && (
              <div className="grid max-h-[62vh] grid-cols-1 gap-3 overflow-y-auto pr-1 sm:grid-cols-2">
                {personas.map((persona) => {
                  const isSelected = selectedPersonaId === persona.id
                  const isPlaying = playingPersonaId === persona.id

                  return (
                    <div
                      className="relative overflow-hidden rounded-lg border bg-card shadow-sm transition hover:border-foreground/30 hover:shadow-md data-[selected=true]:border-foreground"
                      data-selected={isSelected}
                      key={persona.id}
                    >
                      <button
                        aria-pressed={isSelected}
                        className="flex min-h-24 w-full items-center gap-3 p-3 pr-12 pb-12 text-left outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
                        onClick={() => handleSelectPersona(persona.id)}
                        type="button"
                      >
                        <Avatar className="size-14" size="lg">
                          <AvatarImage
                            alt={`${persona.name} profile`}
                            src={
                              persona.profilePictureUrl ??
                              defaultProfilePictureUrl
                            }
                          />
                          <AvatarFallback>
                            {getInitials(persona.name)}
                          </AvatarFallback>
                        </Avatar>
                        <span className="grid min-w-0 flex-1 gap-1">
                          <span className="truncate font-medium">
                            {persona.name}
                          </span>
                          <span className="line-clamp-2 text-muted-foreground">
                            {persona.description || 'No description provided.'}
                          </span>
                        </span>
                      </button>

                      <div className="absolute top-2 right-2 flex flex-col gap-1">
                        {isSelected && (
                          <span className="flex size-7 items-center justify-center rounded-full bg-primary text-primary-foreground">
                            <CheckIcon className="size-4" />
                            <span className="sr-only">Currently selected</span>
                          </span>
                        )}
                        {persona.audioSampleUrl && (
                          <Button
                            aria-label={`${
                              isPlaying ? 'Stop' : 'Play'
                            } ${persona.name} audio sample`}
                            className={
                              isPlaying
                                ? 'text-primary ring-1 ring-primary/30'
                                : undefined
                            }
                            onClick={() => handleAudioPreview(persona)}
                            size="icon-sm"
                            title={
                              isPlaying
                                ? 'Stop audio preview'
                                : 'Play audio preview'
                            }
                            type="button"
                            variant="ghost"
                          >
                            <Music2Icon
                              className={isPlaying ? 'animate-pulse' : undefined}
                            />
                          </Button>
                        )}
                      </div>
                      <div className="absolute right-2 bottom-2">
                        {persona.isDefault ? (
                          <span
                            aria-label="Default personas cannot be edited"
                            className="flex size-8 items-center justify-center rounded-md border bg-muted/50 text-muted-foreground"
                            title="Default personas cannot be edited"
                          >
                            <LockKeyholeIcon className="size-4" />
                          </span>
                        ) : (
                          <Button
                            aria-label={`Edit ${persona.name}`}
                            onClick={() => handleStartEditing(persona)}
                            size="icon-sm"
                            title={`Edit ${persona.name}`}
                            type="button"
                            variant="outline"
                          >
                            <PencilIcon />
                          </Button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </TabsContent>

          <TabsContent className="min-h-0 overflow-y-auto" value="create">
            <form className="grid gap-4" onSubmit={handleCreatePersona}>
              <label className="grid gap-1.5">
                <span className="font-medium">Name</span>
                <Input
                  autoComplete="off"
                  minLength={3}
                  name="name"
                  placeholder="Persona name"
                  required
                />
              </label>

              <label className="grid gap-1.5">
                <span className="font-medium">Description</span>
                <Input
                  autoComplete="off"
                  name="description"
                  placeholder="A short description"
                />
              </label>

              <label className="grid gap-1.5">
                <span className="font-medium">Instruction prompt</span>
                <Textarea
                  className="min-h-28"
                  name="instruction_prompt"
                  placeholder="Describe this persona's personality, behavior, and speaking style."
                  required
                />
              </label>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="grid gap-1.5">
                  <span className="font-medium">Profile picture</span>
                  <Input
                    accept="image/png,image/jpeg,image/webp"
                    className="h-auto py-1"
                    name="profile_picture"
                    type="file"
                  />
                  <span className="text-[10px] text-muted-foreground">
                    Optional PNG, JPG, or WebP.
                  </span>
                </label>

                <label className="grid gap-1.5">
                  <span className="font-medium">Audio sample</span>
                  <Input
                    accept="audio/mpeg,audio/wav,.mp3,.wav"
                    className="h-auto py-1"
                    name="audio_sample"
                    type="file"
                  />
                  <span className="text-[10px] text-muted-foreground">
                    Optional MP3 or WAV.
                  </span>
                </label>
              </div>

              {createError && (
                <p className="text-destructive" role="alert">
                  {createError}
                </p>
              )}

              <div className="flex justify-end">
                <Button disabled={isCreating} type="submit">
                  {isCreating ? <Spinner /> : <PlusIcon />}
                  Create persona
                </Button>
              </div>
            </form>
          </TabsContent>

          {editingPersona && (
            <TabsContent className="min-h-0 overflow-y-auto" value="edit">
              <form
                className="grid gap-4"
                key={`${editingPersona.id}-${editingPersona.updated}`}
                onSubmit={handleEditPersona}
              >
                <label className="grid gap-1.5">
                  <span className="font-medium">Name</span>
                  <Input
                    autoComplete="off"
                    defaultValue={editingPersona.name}
                    minLength={3}
                    name="name"
                    required
                  />
                </label>

                <label className="grid gap-1.5">
                  <span className="font-medium">Description</span>
                  <Input
                    autoComplete="off"
                    defaultValue={editingPersona.description}
                    name="description"
                  />
                </label>

                <label className="grid gap-1.5">
                  <span className="font-medium">Instruction prompt</span>
                  <Textarea
                    className="min-h-28"
                    defaultValue={editingPersona.instructionPrompt}
                    name="instruction_prompt"
                    required
                  />
                </label>

                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="grid gap-1.5">
                    <span className="font-medium">Profile picture</span>
                    <Input
                      accept="image/png,image/jpeg,image/webp"
                      className="h-auto py-1"
                      name="profile_picture"
                      type="file"
                    />
                    {editingPersona.profilePictureUrl && (
                      <img
                        alt={`${editingPersona.name} current profile`}
                        className="size-14 rounded-full border object-cover"
                        src={editingPersona.profilePictureUrl}
                      />
                    )}
                    <span className="text-[10px] text-muted-foreground">
                      {editingPersona.profilePictureUrl
                        ? 'Current image will remain unless replaced.'
                        : 'Optional PNG, JPG, or WebP.'}
                    </span>
                  </label>

                  <label className="grid gap-1.5">
                    <span className="font-medium">Audio sample</span>
                    <Input
                      accept="audio/mpeg,audio/wav,.mp3,.wav"
                      className="h-auto py-1"
                      name="audio_sample"
                      type="file"
                    />
                    {editingPersona.audioSampleUrl && (
                      <audio
                        className="h-8 w-full"
                        controls
                        src={editingPersona.audioSampleUrl}
                      >
                        Current audio sample
                      </audio>
                    )}
                    <span className="text-[10px] text-muted-foreground">
                      {editingPersona.audioSampleUrl
                        ? 'Current audio will remain unless replaced.'
                        : 'Optional MP3 or WAV.'}
                    </span>
                  </label>
                </div>

                {editError && (
                  <p className="text-destructive" role="alert">
                    {editError}
                  </p>
                )}
                {preparationMessage && (
                  <p
                    className={
                      preparationMessage.includes('failed') ||
                      preparationMessage.includes('could not')
                        ? 'text-amber-700 dark:text-amber-400'
                        : 'text-muted-foreground'
                    }
                    role="status"
                  >
                    {preparationId && <Spinner className="mr-2 inline-flex" />}
                    {preparationMessage}
                  </p>
                )}

                <div className="flex justify-end">
                  <Button disabled={isSaving} type="submit">
                    {isSaving ? <Spinner /> : <SaveIcon />}
                    Save changes
                  </Button>
                </div>
              </form>
            </TabsContent>
          )}
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
