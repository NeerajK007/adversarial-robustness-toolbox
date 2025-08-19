from transformers import pipeline
import torch

# This will tell the pipeline to use the CPU
device = "cpu"

# Load the desired model. 'whisper-small' is a good choice for CPU.
# Use a more compact model if you encounter memory issues.
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small", device=device)

# Provide the audio file to transcribe. It can be a local file path.
result = transcriber("harvard.wav")

print(result["text"])
