from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import base64
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import numpy as np
from InferenceInterfaces.ControllableInterface import ControllableInterface
import logging
import sys

# Configure advanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("FastAPI app created successfully")

@app.on_event("startup")
async def startup_event():
    try:
        # Test critical imports
        import torch
        import torchaudio
        from InferenceInterfaces.ControllableInterface import ControllableInterface
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
    except Exception:
        logger.critical("STARTUP FAILED:\n%s", traceback.format_exc())
        raise  # Crash the app if startup fails

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception: %s\nRequest URL: %s\n%s",
        str(exc),
        request.url,
        traceback.format_exc()
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

class TTSParameters(BaseModel):
    text: str
    language: str = "eng"
    duration_scaling_factor: float = 1.0
    prosody_creativity: float = 0.0
    pitch_shift: float = 1.0
    speaking_rate: float = 1.0
    energy_scale: float = 1.0
    emotion_embedding: float = 0.0

def float2pcm(wav: np.ndarray) -> bytes:
    """Convert floating point audio to 16-bit PCM format"""
    wav = np.clip(wav, -1.0, 1.0)
    wav = (wav * 32767).astype(np.int16)
    return wav.tobytes()


@app.get("/")
async def root():
    return {"status": "OK"}

@app.post("/synthesize/")
async def synthesize_speech(params: TTSParameters):
    try:
        # Initialize TTS model
        tts = ControllableInterface()
        
        # Generate speech and visualization
        sr, wav, fig = tts.read(
            prompt=params.text,
            reference_audio=None,
            language=params.language,
            accent=params.language,
            voice_seed=123456,
            prosody_creativity=params.prosody_creativity,
            duration_scaling_factor=params.duration_scaling_factor,
            pause_duration_scaling_factor=1.,
            pitch_variance_scale=1.,
            energy_variance_scale=params.energy_scale,
            emb_slider_1=0.,
            emb_slider_2=0.,
            emb_slider_3=0.,
            emb_slider_4=0.,
            emb_slider_5=0.,
            emb_slider_6=0.,
            loudness_in_db=-24.
        )
        
        # Convert audio to base64
        audio_bytes = float2pcm(wav)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Convert visualization to base64
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        return {
            "audio": {
                "sample_rate": sr,
                "content_base64": audio_b64,
                "format": "audio/wav"
            },
            "visualization": {
                "content_base64": img_b64,
                "format": "image/png"
            }
        }
        
    except Exception as e:
        logger.critical(f"INITIALIZATION FAILED: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up resources
        if 'tts' in locals():
            del tts
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8954)
