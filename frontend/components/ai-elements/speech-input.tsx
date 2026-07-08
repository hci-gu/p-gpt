"use client";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { LanguagesIcon, MicIcon, SquareIcon } from "lucide-react";
import type { ComponentProps } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AsrWorkerRequest,
  AsrWorkerResponse,
  SpeechLanguage,
} from "../../src/workers/asr-worker";

type BrowserWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };

type SpeechInputStatus =
  | "idle"
  | "loading"
  | "recording"
  | "transcribing"
  | "error";

export interface TranscriptionEvent {
  isFinal: boolean;
  language: SpeechLanguage;
  sessionId: string;
  text: string;
}

export type SpeechInputProps = Omit<
  ComponentProps<typeof Button>,
  "onError"
> & {
  defaultLanguage?: SpeechLanguage;
  onTranscriptionChange?: (event: TranscriptionEvent) => void;
  onTranscriptionError?: (error: string) => void;
  onTranscriptionProcessingChange?: (isProcessing: boolean) => void;
  onTranscriptionStart?: (sessionId: string) => void;
};

const targetSampleRate = 16_000;
const partialIntervalMs = 1400;
const rollingWindowSeconds = 12;
const maxFinalSeconds = 120;

const getAudioContextConstructor = () =>
  window.AudioContext ?? (window as BrowserWindow).webkitAudioContext;

const isSpeechInputSupported = () =>
  typeof window !== "undefined" &&
  typeof Worker !== "undefined" &&
  Boolean(getAudioContextConstructor()) &&
  Boolean(navigator.mediaDevices?.getUserMedia);

const createSessionId = () =>
  `asr-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`;

const mergeAudioChunks = (chunks: Float32Array[], maxSampleCount?: number) => {
  const totalSampleCount = chunks.reduce(
    (sum, chunk) => sum + chunk.length,
    0
  );
  const outputSampleCount = Math.min(totalSampleCount, maxSampleCount ?? totalSampleCount);
  const output = new Float32Array(outputSampleCount);
  let writeOffset = outputSampleCount;
  let remaining = outputSampleCount;

  for (let index = chunks.length - 1; index >= 0 && remaining > 0; index -= 1) {
    const chunk = chunks[index];
    const samplesToCopy = Math.min(chunk.length, remaining);
    writeOffset -= samplesToCopy;
    output.set(chunk.slice(chunk.length - samplesToCopy), writeOffset);
    remaining -= samplesToCopy;
  }

  return output;
};

const resampleToTargetRate = (
  audio: Float32Array,
  sourceSampleRate: number
) => {
  if (sourceSampleRate === targetSampleRate) {
    return audio.slice();
  }

  const sampleRateRatio = sourceSampleRate / targetSampleRate;
  const outputLength = Math.max(1, Math.round(audio.length / sampleRateRatio));
  const output = new Float32Array(outputLength);

  for (let index = 0; index < outputLength; index += 1) {
    const sourceIndex = index * sampleRateRatio;
    const beforeIndex = Math.floor(sourceIndex);
    const afterIndex = Math.min(audio.length - 1, beforeIndex + 1);
    const weight = sourceIndex - beforeIndex;
    const before = audio[beforeIndex] ?? 0;
    const after = audio[afterIndex] ?? before;

    output[index] = before + (after - before) * weight;
  }

  return output;
};

export const SpeechInput = ({
  className,
  defaultLanguage = "en",
  disabled,
  onTranscriptionChange,
  onTranscriptionError,
  onTranscriptionProcessingChange,
  onTranscriptionStart,
  ...props
}: SpeechInputProps) => {
  const [language, setLanguage] = useState<SpeechLanguage>(defaultLanguage);
  const [status, setStatus] = useState<SpeechInputStatus>("idle");
  const [isSupported] = useState(isSpeechInputSupported);
  const audioContextRef = useRef<AudioContext | null>(null);
  const fullChunksRef = useRef<Float32Array[]>([]);
  const isRecordingRef = useRef(false);
  const lastPartialSentAtRef = useRef(0);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const rollingChunksRef = useRef<Float32Array[]>([]);
  const rollingSampleCountRef = useRef(0);
  const sessionIdRef = useRef<string | null>(null);
  const sourceSampleRateRef = useRef(targetSampleRate);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const onTranscriptionChangeRef =
    useRef<SpeechInputProps["onTranscriptionChange"]>(onTranscriptionChange);
  const onTranscriptionErrorRef =
    useRef<SpeechInputProps["onTranscriptionError"]>(onTranscriptionError);
  const onTranscriptionProcessingChangeRef = useRef<
    SpeechInputProps["onTranscriptionProcessingChange"]
  >(onTranscriptionProcessingChange);
  const onTranscriptionStartRef =
    useRef<SpeechInputProps["onTranscriptionStart"]>(onTranscriptionStart);

  onTranscriptionChangeRef.current = onTranscriptionChange;
  onTranscriptionErrorRef.current = onTranscriptionError;
  onTranscriptionProcessingChangeRef.current = onTranscriptionProcessingChange;
  onTranscriptionStartRef.current = onTranscriptionStart;

  const postWorkerMessage = useCallback((message: AsrWorkerRequest) => {
    if (message.type === "transcribe") {
      workerRef.current?.postMessage(message, [
        message.audio.buffer as ArrayBuffer,
      ]);
      return;
    }

    workerRef.current?.postMessage(message);
  }, []);

  const getWorker = useCallback(() => {
    if (workerRef.current) {
      return workerRef.current;
    }

    const worker = new Worker(
      new URL("../../src/workers/asr-worker.ts", import.meta.url),
      { type: "module" }
    );

    worker.addEventListener("message", (event: MessageEvent<AsrWorkerResponse>) => {
      const message = event.data;

      if (message.type === "ready") {
        if (!isRecordingRef.current) {
          setStatus("idle");
        }
        return;
      }

      if (message.type === "error") {
        setStatus("error");
        onTranscriptionProcessingChangeRef.current?.(false);
        onTranscriptionErrorRef.current?.(message.error);
        return;
      }

      if (message.sessionId !== sessionIdRef.current) {
        return;
      }

      onTranscriptionChangeRef.current?.({
        isFinal: message.isFinal,
        language: message.language,
        sessionId: message.sessionId,
        text: message.text,
      });

      if (message.isFinal) {
        setStatus("idle");
      }
    });

    workerRef.current = worker;
    return worker;
  }, []);

  const sendTranscriptionWindow = useCallback(
    (isFinal: boolean) => {
      const sessionId = sessionIdRef.current;
      const sourceSampleRate = sourceSampleRateRef.current;
      const chunks = isFinal ? fullChunksRef.current : rollingChunksRef.current;

      if (!sessionId || chunks.length === 0) {
        return;
      }

      const sourceAudio = mergeAudioChunks(
        chunks,
        isFinal ? sourceSampleRate * maxFinalSeconds : undefined
      );
      const audio = resampleToTargetRate(sourceAudio, sourceSampleRate);

      postWorkerMessage({
        audio,
        isFinal,
        language,
        sampleRate: targetSampleRate,
        sessionId,
        type: "transcribe",
      });
    },
    [language, postWorkerMessage]
  );

  const cleanupAudio = useCallback(async () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }

    sourceRef.current?.disconnect();
    sourceRef.current = null;

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    if (!isSupported || status === "loading") {
      return;
    }

    const AudioContextConstructor = getAudioContextConstructor();

    if (!AudioContextConstructor) {
      setStatus("error");
      return;
    }

    setStatus("loading");
    onTranscriptionProcessingChangeRef.current?.(false);

    try {
      const worker = getWorker();
      const sessionId = createSessionId();
      sessionIdRef.current = sessionId;
      worker.postMessage({ language, type: "init" } satisfies AsrWorkerRequest);
      onTranscriptionStartRef.current?.(sessionId);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const audioContext = new AudioContextConstructor();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      streamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceRef.current = source;
      processorRef.current = processor;
      sourceSampleRateRef.current = audioContext.sampleRate;
      fullChunksRef.current = [];
      rollingChunksRef.current = [];
      rollingSampleCountRef.current = 0;
      lastPartialSentAtRef.current = 0;
      isRecordingRef.current = true;

      processor.onaudioprocess = (event) => {
        if (!isRecordingRef.current) {
          return;
        }

        const input = event.inputBuffer.getChannelData(0);
        const chunk = new Float32Array(input);
        const sourceSampleRate = sourceSampleRateRef.current;
        const rollingMaxSamples = sourceSampleRate * rollingWindowSeconds;

        fullChunksRef.current.push(chunk);
        rollingChunksRef.current.push(chunk);
        rollingSampleCountRef.current += chunk.length;

        while (
          rollingSampleCountRef.current > rollingMaxSamples &&
          rollingChunksRef.current.length > 1
        ) {
          const removedChunk = rollingChunksRef.current.shift();
          rollingSampleCountRef.current -= removedChunk?.length ?? 0;
        }

        const now = performance.now();

        if (now - lastPartialSentAtRef.current >= partialIntervalMs) {
          lastPartialSentAtRef.current = now;
          sendTranscriptionWindow(false);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      await audioContext.resume();
      setStatus("recording");
    } catch (error) {
      isRecordingRef.current = false;
      sessionIdRef.current = null;
      await cleanupAudio();
      setStatus("error");
      onTranscriptionProcessingChangeRef.current?.(false);
      onTranscriptionErrorRef.current?.(
        error instanceof Error ? error.message : "Could not start recording."
      );
    }
  }, [cleanupAudio, getWorker, isSupported, language, sendTranscriptionWindow, status]);

  const stopRecording = useCallback(async () => {
    if (!isRecordingRef.current) {
      return;
    }

    isRecordingRef.current = false;
    setStatus("transcribing");
    onTranscriptionProcessingChangeRef.current?.(true);
    sendTranscriptionWindow(true);
    await cleanupAudio();
  }, [cleanupAudio, sendTranscriptionWindow]);

  const toggleRecording = useCallback(() => {
    if (status === "recording") {
      void stopRecording();
      return;
    }

    void startRecording();
  }, [startRecording, status, stopRecording]);

  const toggleLanguage = useCallback(() => {
    setLanguage((currentLanguage) => {
      const nextLanguage = currentLanguage === "en" ? "sv" : "en";
      postWorkerMessage({ language: nextLanguage, type: "setLanguage" });
      return nextLanguage;
    });
  }, [postWorkerMessage]);

  useEffect(
    () => () => {
      isRecordingRef.current = false;
      void cleanupAudio();
      workerRef.current?.terminate();
      workerRef.current = null;
    },
    [cleanupAudio]
  );

  const isLoading = status === "loading" || status === "transcribing";
  const isRecording = status === "recording";
  const isDisabled = disabled || !isSupported || status === "error" || isLoading;

  return (
    <div className="inline-flex items-center gap-0.5">
      <Button
        aria-label={`Switch transcription language, currently ${
          language === "en" ? "English" : "Swedish"
        }`}
        className="h-8 gap-1 px-2 text-xs uppercase"
        disabled={disabled || isRecording || isLoading || !isSupported}
        onClick={toggleLanguage}
        type="button"
        variant="ghost"
      >
        <LanguagesIcon className="size-3.5" />
        {language}
      </Button>
      <div className="relative inline-flex items-center justify-center">
        {isRecording &&
          [0, 1, 2].map((index) => (
            <div
              className="absolute inset-0 animate-ping rounded-full border-2 border-red-400/30"
              key={index}
              style={{
                animationDelay: `${index * 0.3}s`,
                animationDuration: "2s",
              }}
            />
          ))}
        <Button
          aria-label={isRecording ? "Stop transcription" : "Start transcription"}
          className={cn(
            "relative z-10 rounded-full transition-all duration-300",
            isRecording &&
              "bg-destructive text-white hover:bg-destructive/80 hover:text-white",
            className
          )}
          disabled={isDisabled}
          onClick={toggleRecording}
          type="button"
          {...props}
        >
          {isLoading && <Spinner />}
          {!isLoading && isRecording && <SquareIcon className="size-4" />}
          {!(isLoading || isRecording) && <MicIcon className="size-4" />}
        </Button>
      </div>
    </div>
  );
};
