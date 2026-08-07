from fastapi import APIRouter, WebSocket

from config import settings
from services import speaker_runtime


router = APIRouter()


@router.websocket("/speaker/v1")
async def speaker_websocket(websocket: WebSocket) -> None:
    from speaker import SpeakerServices, SpeakerSession

    offered_protocols = {
        protocol.strip()
        for protocol in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if protocol.strip()
    }
    if "p-gpt-speaker.v1" not in offered_protocols:
        await websocket.close(code=1002)
        return
    await websocket.accept(subprotocol="p-gpt-speaker.v1")
    app = websocket.app
    session = SpeakerSession(
        websocket,
        SpeakerServices(
            configure=lambda event: speaker_runtime.configure_session(app, event),
            transcribe=lambda audio, language: speaker_runtime.transcribe_audio(app, audio, language),
            stream_text=speaker_runtime.stream_text,
            synthesize=lambda context, sentence, metadata: speaker_runtime.synthesize_sentence(app, context, sentence, metadata),
        ),
        speaker_runtime.logger,
        reopen_grace_seconds=settings.speaker_reopen_grace_seconds,
    )
    await session.run()
