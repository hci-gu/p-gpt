import { Button } from '@/components/ui/button'
import type { SpeechLanguage } from '@/src/lib/speech-language'
import { LanguagesIcon } from 'lucide-react'

export const SpeechLanguageToggle = ({
  disabled = false,
  language,
  onLanguageChange,
}: {
  disabled?: boolean
  language: SpeechLanguage
  onLanguageChange: (language: SpeechLanguage) => void
}) => (
  <Button
    aria-label={`Switch transcription language, currently ${
      language === 'en' ? 'English' : 'Swedish'
    }`}
    className="h-8 gap-1 border border-border/25 bg-accent/10 px-2 text-xs uppercase shadow-[0_1px_2px_hsl(0_0%_0%/0.04)] hover:bg-accent"
    disabled={disabled}
    onClick={() => onLanguageChange(language === 'en' ? 'sv' : 'en')}
    title={`Transcription language: ${
      language === 'en' ? 'English' : 'Swedish'
    }`}
    type="button"
    variant="ghost"
  >
    <LanguagesIcon className="size-3.5" />
    {language}
  </Button>
)
