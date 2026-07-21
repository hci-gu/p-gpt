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
import type { PersonaRecord } from '@/lib/personas'
import { useChatStore } from '@/src/state/chat'
import { usePersonasStore } from '@/src/state/personas'
import { CheckIcon, Music2Icon, PlusIcon, UsersIcon } from 'lucide-react'
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
  const selectPersona = usePersonasStore((state) => state.selectPersona)
  const selectPersonaForNewChat = useChatStore(
    (state) => state.selectPersonaForNewChat
  )
  const [activeTab, setActiveTab] = useState('choose')
  const [playingPersonaId, setPlayingPersonaId] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
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
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="choose">
              <UsersIcon />
              Choose persona
            </TabsTrigger>
            <TabsTrigger value="create">
              <PlusIcon />
              Create persona
            </TabsTrigger>
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
                        className="flex min-h-24 w-full items-center gap-3 p-3 pr-12 text-left outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
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
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
