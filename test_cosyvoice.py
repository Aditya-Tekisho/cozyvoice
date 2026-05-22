import os
import torch
import soundfile as sf

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

MODEL_PATH = "pretrained_models/CosyVoice-300M"

cosyvoice = CosyVoice(MODEL_PATH)

prompt_speech_16k = load_wav("sample.wav", 16000)

output = cosyvoice.inference_zero_shot(
    "Hello, this is a test voice cloning demo.",
    "This is the original speaker voice.",
    prompt_speech_16k
)

os.makedirs("outputs", exist_ok=True)

for i, result in enumerate(output):
    sf.write(
        f"outputs/output_{i}.wav",
        result["tts_speech"],
        22050
    )

print("DONE")