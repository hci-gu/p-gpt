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
  PromptInputButton,
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
import { Suggestions } from '@/components/ai-elements/suggestion'
import { GlobeIcon } from 'lucide-react'
import type { ChangeEvent } from 'react'
import { useCallback, useMemo } from 'react'
import { suggestions, useChatStore } from '../../state/chat'
import {
  AudioMessage,
  PromptInputAttachmentsDisplay,
  SuggestionItem,
} from './components'

const ChatPage = () => {
  const text = useChatStore((state) => state.text)
  const useWebSearch = useChatStore((state) => state.useWebSearch)
  const status = useChatStore((state) => state.status)
  const messages = useChatStore((state) => state.messages)
  const setText = useChatStore((state) => state.setText)
  const appendTranscription = useChatStore((state) => state.appendTranscription)
  const toggleWebSearch = useChatStore((state) => state.toggleWebSearch)
  const submitMessage = useChatStore((state) => state.submitMessage)
  const completeAssistantResponse = useChatStore(
    (state) => state.completeAssistantResponse
  )
  const failAssistantResponse = useChatStore(
    (state) => state.failAssistantResponse
  )

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
    (transcript: string) => {
      appendTranscription(transcript)
    },
    [appendTranscription]
  )

  const handleTextChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      setText(event.target.value)
    },
    [setText]
  )

  const isSubmitDisabled = useMemo(
    () => !text.trim() || status === 'submitted' || status === 'streaming',
    [text, status]
  )

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
                        {version.audioUrl ? (
                          <AudioMessage
                            onEnded={() => {
                              completeAssistantResponse(version.id)
                            }}
                            onError={() => {
                              failAssistantResponse(version.id)
                            }}
                            src={version.audioUrl}
                          />
                        ) : (
                          <MessageResponse>{version.content}</MessageResponse>
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
        <Suggestions className="px-4">
          {suggestions.map((suggestion) => (
            <SuggestionItem
              key={suggestion}
              onClick={handleSuggestionClick}
              suggestion={suggestion}
            />
          ))}
        </Suggestions>
        <div className="w-full px-4 pb-4">
          <PromptInput globalDrop multiple onSubmit={handleSubmit}>
            <PromptInputHeader>
              <PromptInputAttachmentsDisplay />
            </PromptInputHeader>
            <PromptInputBody>
              <PromptInputTextarea onChange={handleTextChange} value={text} />
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
                  onTranscriptionChange={handleTranscriptionChange}
                  size="icon-sm"
                  variant="ghost"
                />
                <PromptInputButton
                  onClick={toggleWebSearch}
                  variant={useWebSearch ? 'default' : 'ghost'}
                >
                  <GlobeIcon size={16} />
                  <span>Search</span>
                </PromptInputButton>
              </PromptInputTools>
              <PromptInputSubmit disabled={isSubmitDisabled} status={status} />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
