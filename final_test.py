from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import numpy as np
import sounddevice as sd
import threading
from pynput import keyboard
from pydub import AudioSegment
import io
import math
import librosa
import soundfile as sf

fs = 44100  # Sample rate (44.1kHz is CD quality)
audio_data = []  # List to store recorded chunks
is_paused = threading.Event()  # Event to control pause
is_recording = True  # Flag to control the recording loop


def record_audio():
    global audio_data, is_recording, is_paused

    print("Recording started. Press 'p' to pause, 'r' to resume, and 'e' to end the recording.")
    while is_recording:
        if not is_paused.is_set():  # Check if not paused
            # Record a short chunk of audio (100ms)
            chunk = sd.rec(int(0.1 * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Wait for the chunk to finish recording
            audio_data.append(chunk)  # Append recorded chunk to the list
            
        sd.sleep(100)  # Sleep for 100ms before next check (smooth loop)

    # Concatenate the chunks when the recording ends
    audio_data_np = np.concatenate(audio_data, axis=0)
    
    # Save the final recording to an in-memory buffer
    buffer = io.BytesIO()
    
    # Use soundfile to write the numpy array into a .wav buffer
    sf.write(buffer, audio_data_np.astype(np.float32), fs, format='WAV')
    buffer.seek(0)
    
    print("\nRecording ended.")
    
    return buffer


def start_recording():
    global is_recording
    is_recording = True
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

def pause_recording():
    global is_paused
    is_paused.set()
    print("Recording paused. Press 'r' to resume.")

def resume_recording():
    global is_paused
    is_paused.clear()
    print("Recording resumed. Press 'p' to pause or 'e' to end.")

def end_recording():
    global is_recording
    is_recording = False
    print("\nEnding recording...")

def on_press(key):
    try:
        if key.char == 'p':
            pause_recording()
        elif key.char == 'r':
            resume_recording()
        elif key.char == 'e':
            end_recording()
            return False  # Stop listener
    except AttributeError:
        pass

# Start the recording processe
start_recording()

# Setup keyboard listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

# Load the audio from the in-memory buffer
buffer = record_audio()
audio = AudioSegment.from_wav(buffer)

# Define chunk length in milliseconds (3 seconds = 3000 ms)
chunk_length_ms = 3000

# Calculate number of chunks
num_chunks = math.ceil(len(audio) / chunk_length_ms)

# Process each chunk from the in-memory audio data
predicted_labels = []
processor = Wav2Vec2Processor.from_pretrained("/Users/mac/Documents/speech_model/trained_model")
model = Wav2Vec2ForSequenceClassification.from_pretrained("/Users/mac/Documents/speech_model/trained_model")

for i in range(num_chunks):
    start = i * chunk_length_ms
    end = min((i + 1) * chunk_length_ms, len(audio))
    
    # Extract the chunk
    chunk = audio[start:end]
    
    # Convert the chunk to an in-memory buffer
    chunk_buffer = io.BytesIO()
    chunk.export(chunk_buffer, format="wav")
    chunk_buffer.seek(0)
    
    # Load the chunk
    speech, sr = librosa.load(chunk_buffer, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Forward pass to get the logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label (index of the highest logit value)
    predicted_label = torch.argmax(logits, dim=-1).item()
    predicted_labels.append(predicted_label)

emotion_mapping = {
    0: "fear",
    1: "angry",
    2: "disgust",
    3: "neutral",
    4: "sad",
    5: "pleasant surprise",
    6: "happy"
}

def emotion_percentage(predicted_labels, emotion_mapping):
    # Map each predicted label to its corresponding emotion
    emotion_labels = [emotion_mapping[label] for label in predicted_labels]
    
    # Calculate total number of predictions
    total = len(emotion_labels)
    
    # Create a dictionary to store the count of each emotion
    emotion_count = {emotion: emotion_labels.count(emotion) for emotion in set(emotion_labels)}
    
    # Calculate and print percentage for each emotion
    emotion_percentage = {emotion: (count / total) * 100 for emotion, count in emotion_count.items()}
    
    return emotion_percentage

# Get the percentage of each emotion
percentages = emotion_percentage(predicted_labels, emotion_mapping)

# Print the results
for emotion, percentage in percentages.items():
    print(f"{emotion}: {percentage:.2f}%")
