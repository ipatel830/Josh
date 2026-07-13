import shutil
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import josh_ai as j_ai

app = FastAPI(title="Josh Voice Assistant")

# --- Load models once, at startup, not per-request ---
NLU_PATH = "nlu/nlu_model.pt"
WHISPER_PATH = "whisper-tiny-local/"

assistant = None


@app.on_event("startup")
def load_pipeline():
    global assistant
    assistant = j_ai.VoiceAssistantPipeline(
        nlu_path=NLU_PATH,
        whisper_path=WHISPER_PATH
    )


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the container/model is up and ready."""
    return {"status": "ok", "model_loaded": assistant is not None}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Accepts an uploaded audio file, runs it through the full
    STT -> NLU pipeline, and returns the structured result.
    """
    if assistant is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    suffix = os.path.splitext(audio.filename)[1] or ".wav"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        result = assistant.run(tmp_path)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)