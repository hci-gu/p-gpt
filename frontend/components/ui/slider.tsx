import * as React from "react"
import { Slider as SliderPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

function Slider({
  className,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root
      className={cn(
        "relative flex w-full touch-none select-none items-center data-disabled:cursor-not-allowed data-disabled:opacity-50",
        className
      )}
      data-slot="slider"
      {...props}
    >
      <SliderPrimitive.Track
        className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-input dark:bg-input/80"
        data-slot="slider-track"
      >
        <SliderPrimitive.Range
          className="absolute h-full bg-primary"
          data-slot="slider-range"
        />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        className="block size-3.5 shrink-0 rounded-full border border-primary bg-background shadow-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:pointer-events-none"
        data-slot="slider-thumb"
      />
    </SliderPrimitive.Root>
  )
}

export { Slider }
