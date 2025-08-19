import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np

# Model and audio settings
model_size = "tiny.en"  # or "base", "small" for better accuracy
# Run on CPU with INT8 computation for efficiency
model = WhisperModel(model_size, device="cpu", compute_type="int8")
sample_rate = 16000  # Whisper models are trained on 16kHz audio
block_size = 1024  # Size of each audio chunk

print("Starting transcription...")
# This function will be called for each audio block
def callback(indata, frames, time, status):
    if status:
        print(status)

    # Pass the audio chunk to the model for transcription
    segments, info = model.transcribe(indata.flatten(), beam_size=5)

    # Iterate over the transcribed segments and print the text
    for segment in segments:
        print(f"[Real-time] {segment.text}")

# Set up and start the audio stream
with sd.InputStream(samplerate=sample_rate, blocksize=block_size, callback=callback, dtype='int16', channels=1) as stream:
    print("Listening... Press Ctrl+C to stop.")
    try:
        # Keep the program running indefinitely
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nTranscription stopped.")
        
        