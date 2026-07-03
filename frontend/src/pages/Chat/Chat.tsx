'use client'

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation'
import {
  Message,
  MessageBranch,
  MessageBranchContent,
  MessageBranchNext,
  MessageBranchPage,
  MessageBranchPrevious,
  MessageBranchSelector,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message'
import type { PromptInputMessage } from '@/components/ai-elements/prompt-input'
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputBody,
  PromptInputFooter,
  PromptInputHeader,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from '@/components/ai-elements/prompt-input'
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning'
import {
  Source,
  Sources,
  SourcesContent,
  SourcesTrigger,
} from '@/components/ai-elements/sources'
import { SpeechInput } from '@/components/ai-elements/speech-input'
import type { TranscriptionEvent } from '@/components/ai-elements/speech-input'
import { Suggestions } from '@/components/ai-elements/suggestion'
import { Spinner } from '@/components/ui/spinner'
import type { ChangeEvent } from 'react'
import { useCallback, useMemo, useRef, useState } from 'react'
import { suggestions, useChatStore } from '../../state/chat'
import {
  AnimatedMessageResponse,
  AudioMessage,
  PromptInputAttachmentsDisplay,
  SuggestionItem,
} from './components'

const getAssistantFallbackText = (
  contentStatus: 'pending' | 'ready' | 'error' | undefined
) => {
  if (contentStatus === 'error') {
    return 'Failed to generate audio response.'
  }

  return 'Generating audio...'
}

const transcriptionRevealDurationMs = 900

const splitIntoRevealTokens = (content: string) =>
  content.match(/\S+\s*/g) ?? (content ? [content] : [])

const ChatPage = () => {
  const text = useChatStore((state) => state.text)
  const status = useChatStore((state) => state.status)
  const messages = useChatStore((state) => state.messages)
  const setText = useChatStore((state) => state.setText)
  const beginTranscriptionDraft = useChatStore(
    (state) => state.beginTranscriptionDraft
  )
  const updateTranscriptionDraft = useChatStore(
    (state) => state.updateTranscriptionDraft
  )
  const finishTranscriptionDraft = useChatStore(
    (state) => state.finishTranscriptionDraft
  )
  const submitMessage = useChatStore((state) => state.submitMessage)
  const completeAssistantResponse = useChatStore(
    (state) => state.completeAssistantResponse
  )
  const failAssistantResponse = useChatStore(
    (state) => state.failAssistantResponse
  )
  const interruptAssistantResponse = useChatStore(
    (state) => state.interruptAssistantResponse
  )
  const [isTranscribing, setIsTranscribing] = useState(false)
  const transcriptionAnimationRef = useRef<number | null>(null)

  const handleSubmit = useCallback(
    (message: PromptInputMessage) => {
      const hasText = Boolean(message.text)
      const hasAttachments = Boolean(message.files?.length)

      if (!(hasText || hasAttachments)) {
        return
      }

      submitMessage(message.text || 'Sent with attachments')
    },
    [submitMessage]
  )

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      submitMessage(suggestion)
    },
    [submitMessage]
  )

  const handleTranscriptionChange = useCallback(
    (event: TranscriptionEvent) => {
      if (!event.isFinal) {
        return
      }

      const tokens = splitIntoRevealTokens(event.text.trim())

      if (transcriptionAnimationRef.current) {
        cancelAnimationFrame(transcriptionAnimationRef.current)
      }

      if (tokens.length === 0) {
        finishTranscriptionDraft(event.sessionId, '')
        setIsTranscribing(false)
        return
      }

      const startedAt = performance.now()

      const tick = (now: number) => {
        const progress = Math.min(
          1,
          (now - startedAt) / transcriptionRevealDurationMs
        )
        const visibleTokenCount = Math.max(
          1,
          Math.ceil(progress * tokens.length)
        )
        const visibleText = tokens.slice(0, visibleTokenCount).join('')

        updateTranscriptionDraft(event.sessionId, visibleText)

        if (progress < 1) {
          transcriptionAnimationRef.current = requestAnimationFrame(tick)
          return
        }

        finishTranscriptionDraft(event.sessionId, event.text)
        transcriptionAnimationRef.current = null
        setIsTranscribing(false)
      }

      transcriptionAnimationRef.current = requestAnimationFrame(tick)
    },
    [finishTranscriptionDraft, updateTranscriptionDraft]
  )

  const handleTranscriptionStart = useCallback(
    (sessionId: string) => {
      if (transcriptionAnimationRef.current) {
        cancelAnimationFrame(transcriptionAnimationRef.current)
        transcriptionAnimationRef.current = null
      }
      setIsTranscribing(false)
      beginTranscriptionDraft(sessionId)
    },
    [beginTranscriptionDraft]
  )

  const handleTranscriptionProcessingChange = useCallback(
    (isProcessing: boolean) => {
      setIsTranscribing(isProcessing)
    },
    []
  )

  const handleTextChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      setText(event.target.value)
    },
    [setText]
  )

  const isGenerating = status === 'submitted' || status === 'streaming'
  const isSubmitDisabled = useMemo(
    () => !isGenerating && (!text.trim() || isTranscribing),
    [isGenerating, isTranscribing, text]
  )
  const shouldShowSuggestions = messages.length === 0

  return (
    <div className="relative flex size-full flex-col divide-y overflow-hidden">
      <Conversation>
        <ConversationContent>
          {messages.map(({ versions, ...message }) => (
            <MessageBranch defaultBranch={0} key={message.key}>
              <MessageBranchContent>
                {versions.map((version) => (
                  <Message
                    from={message.from}
                    key={`${message.key}-${version.id}`}
                  >
                    <div>
                      {message.sources?.length && (
                        <Sources>
                          <SourcesTrigger count={message.sources.length} />
                          <SourcesContent>
                            {message.sources.map((source) => (
                              <Source
                                href={source.href}
                                key={source.href}
                                title={source.title}
                              />
                            ))}
                          </SourcesContent>
                        </Sources>
                      )}
                      {message.reasoning && (
                        <Reasoning duration={message.reasoning.duration}>
                          <ReasoningTrigger />
                          <ReasoningContent>
                            {message.reasoning.content}
                          </ReasoningContent>
                        </Reasoning>
                      )}
                      <MessageContent>
                        {version.audioUrl &&
                        !version.audioPlaybackComplete &&
                        version.contentStatus !== 'error' ? (
                          <AudioMessage
                            onEnded={() => {
                              completeAssistantResponse(version.id)
                            }}
                            onError={() => {
                              failAssistantResponse(version.id)
                            }}
                            src={version.audioUrl}
                          />
                        ) : message.from === 'assistant' &&
                          version.audioPlaybackComplete &&
                          version.contentStatus === 'ready' ? (
                          <AnimatedMessageResponse content={version.content} />
                        ) : (
                          <MessageResponse>
                            {version.content ||
                              getAssistantFallbackText(version.contentStatus)}
                          </MessageResponse>
                        )}
                      </MessageContent>
                    </div>
                  </Message>
                ))}
              </MessageBranchContent>
              {versions.length > 1 && (
                <MessageBranchSelector>
                  <MessageBranchPrevious />
                  <MessageBranchPage />
                  <MessageBranchNext />
                </MessageBranchSelector>
              )}
            </MessageBranch>
          ))}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
      <div className="grid shrink-0 gap-4 pt-4">
        {shouldShowSuggestions && (
          <Suggestions className="px-4">
            {suggestions.map((suggestion) => (
              <SuggestionItem
                key={suggestion}
                onClick={handleSuggestionClick}
                suggestion={suggestion}
              />
            ))}
          </Suggestions>
        )}
        <div className="w-full px-4 pb-4">
          <PromptInput
            className="[&_[data-slot=input-group]]:bg-[hsl(0_0%_100%_/_var(--prompt-input-surface-opacity))] [&_[data-slot=input-group]]:backdrop-blur-md"
            globalDrop
            multiple
            onSubmit={handleSubmit}
          >
            <PromptInputHeader>
              <PromptInputAttachmentsDisplay />
            </PromptInputHeader>
            <PromptInputBody>
              <div className="relative w-full">
                <PromptInputTextarea
                  className="min-h-24 content-start text-left align-top"
                  disabled={isTranscribing}
                  onChange={handleTextChange}
                  value={text}
                />
                {isTranscribing && (
                  <div className="absolute inset-0 flex items-center justify-center rounded-md bg-background/50 backdrop-blur-[1px]">
                    <Spinner />
                  </div>
                )}
              </div>
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger />
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>
                <SpeechInput
                  className="shrink-0"
                  defaultLanguage="en"
                  onTranscriptionChange={handleTranscriptionChange}
                  onTranscriptionProcessingChange={
                    handleTranscriptionProcessingChange
                  }
                  onTranscriptionStart={handleTranscriptionStart}
                  size="icon-sm"
                  variant="ghost"
                />
              </PromptInputTools>
              <PromptInputSubmit
                disabled={isSubmitDisabled}
                onStop={interruptAssistantResponse}
                status={status}
              />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
