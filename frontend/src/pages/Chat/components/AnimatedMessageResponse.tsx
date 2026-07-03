'use client'

import { MessageResponse } from '@/components/ai-elements/message'
import { useEffect, useMemo, useState } from 'react'

const defaultRevealDurationMs = 2500

const splitIntoRevealTokens = (content: string) =>
  content.match(/\S+\s*/g) ?? (content ? [content] : [])

export const AnimatedMessageResponse = ({
  content,
  durationMs = defaultRevealDurationMs,
}: {
  content: string
  durationMs?: number
}) => {
  const tokens = useMemo(() => splitIntoRevealTokens(content), [content])
  const [visibleTokenCount, setVisibleTokenCount] = useState(0)

  useEffect(() => {
    if (!content || tokens.length === 0) {
      setVisibleTokenCount(0)
      return undefined
    }

    let animationFrame = 0
    const startedAt = performance.now()

    const tick = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / durationMs)
      const nextVisibleTokenCount = Math.max(
        1,
        Math.ceil(progress * tokens.length)
      )

      setVisibleTokenCount(nextVisibleTokenCount)

      if (progress < 1) {
        animationFrame = requestAnimationFrame(tick)
      }
    }

    animationFrame = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(animationFrame)
    }
  }, [content, durationMs, tokens])

  const visibleContent = tokens.slice(0, visibleTokenCount).join('')
  const isAnimating = visibleTokenCount < tokens.length

  return (
    <div className="animate-in fade-in-0 slide-in-from-bottom-1 duration-300">
      <MessageResponse isAnimating={isAnimating}>
        {visibleContent}
      </MessageResponse>
    </div>
  )
}
