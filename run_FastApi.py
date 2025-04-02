from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import numpy as np
from InferenceInterfaces.ControllableInterface import ControllableInterface

app = FastAPI()

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

@app.post("/synthesize/")
async def synthesize_speech(params: TTSParameters):
    try:
        # Initialize TTS model
        tts = ControllableInterface()
        
        # Generate speech and visualization
        sr, wav, fig = tts.read(
            text=params.text,
            filename=None,
            source_lang=params.language,
            target_lang=params.language,
            speaker_reference=None,
            prosody_creativity=params.prosody_creativity,
            duration_scaling_factor=params.duration_scaling_factor,
            pitch_shift=params.pitch_shift,
            speaking_rate=params.speaking_rate,
            energy_scale=params.energy_scale,
            emotion_embedding=params.emotion_embedding
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
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up resources
        if 'tts' in locals():
            del tts
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8954)
