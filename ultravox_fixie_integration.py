import asyncio
import logging
import threading
import time
import queue
import numpy as np
import pyaudio
import sounddevice as sd
import webrtcvad
from collections import deque

from ultravox.inference import ultravox_infer
from ultravox.data.data_sample import VoiceSample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms
VAD_MODE = 3  # Aggressiveness mode (0-3)

# Initialize Fixie ultravox model
MODEL_PATH = "fixie-ai/ultravox-v0_3-llama-3_2-1b"
inference = ultravox_infer.UltravoxInference(
    MODEL_PATH,
    conversation_mode=True
)

# Initialize PyAudio player
player = pyaudio.PyAudio().open(
    format=pyaudio.paInt16,
    channels=1,
    rate=24000,
    output=True,
    frames_per_buffer=1024
)

# Thread-safe queue for audio playback
playback_queue = queue.Queue()

# Voice Activity Detector
vad = webrtcvad.Vad(VAD_MODE)

# Conversation history
conversation_history = deque(maxlen=10)
EXIT_COMMANDS = {"goodbye", "exit", "quit", "stop"}

def playback_worker():
    """Worker thread to handle audio playback from the queue."""
    logger.info("Audio playback thread started.")
    while True:
        chunk = playback_queue.get()
        if chunk is None:  # Shutdown signal
            logger.info("Audio playback thread received shutdown signal.")
            break
        try:
            player.write(chunk)
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
        finally:
            playback_queue.task_done()
    logger.info("Audio playback thread terminated.")

async def record_audio_vad():
    """Records audio using Voice Activity Detection."""
    logger.info("Recording started. Speak into the microphone...")
    audio_frames = []
    frame_length = int(SAMPLE_RATE * FRAME_DURATION / 1000)

    # VAD parameters
    padding_duration_ms = 200
    num_padding_frames = padding_duration_ms // FRAME_DURATION
    ring_buffer = deque(maxlen=num_padding_frames)
    triggered = False
    max_silence_duration = 0.8  # seconds
    max_recording_duration = 15.0  # seconds
    silence_start_time = None
    recording_start_time = None

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=frame_length,
        dtype='int16',
        channels=CHANNELS,
        latency='low'
    ) as stream:
        while True:
            try:
                data, _ = stream.read(frame_length)
                if not triggered:
                    ring_buffer.append(data)
                    num_voiced = sum(1 for f in ring_buffer if vad.is_speech(f, SAMPLE_RATE))
                    if num_voiced / len(ring_buffer) > 0.5:
                        triggered = True
                        recording_start_time = time.time()
                        audio_frames.extend(ring_buffer)
                        ring_buffer.clear()
                        logger.info("Speech detected. Recording...")
                else:
                    audio_frames.append(data)
                    ring_buffer.append(data)
                    num_unvoiced = sum(0 for f in ring_buffer if vad.is_speech(f, SAMPLE_RATE))
                    silence_ratio = num_unvoiced / len(ring_buffer)

                    if silence_ratio > 0.5 and silence_start_time is None:
                        silence_start_time = time.time()
                    elif silence_ratio <= 0.5 and silence_start_time is not None:
                        silence_start_time = None

                    if silence_start_time and (time.time() - silence_start_time) > max_silence_duration:
                        logger.info("Speech ended after silence.")
                        break

                    if recording_start_time and (time.time() - recording_start_time) > max_recording_duration:
                        logger.info("Max recording duration reached.")
                        break

            except Exception as e:
                logger.warning(f"Error reading audio: {e}")
                continue

    audio_data = b''.join(audio_frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

async def process_tts(text):
    """Process TTS using Fixie's built-in capabilities."""
    sample = VoiceSample.from_prompt(text)
    output = inference.infer_stream(sample)

    for chunk in output:
        if hasattr(chunk, 'audio'):
            playback_queue.put(chunk.audio)
        elif hasattr(chunk, 'text'):
            logger.info(f"TTS response: {chunk.text}")

async def main():
    logger.info("Starting Fixie ultravox voice assistant...")

    # Start playback thread
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    total_cycles = 0
    total_response_time = 0

    while True:
        cycle_start = time.time()
        logger.info("Awaiting user input...")

        # Record audio
        audio_data = await record_audio_vad()
        if audio_data is None:
            continue

        # Create voice sample
        sample = VoiceSample.from_prompt_and_raw("", audio_data, SAMPLE_RATE)

        # Get response
        response = ""
        for chunk in inference.infer_stream(sample):
            if hasattr(chunk, 'text'):
                response += chunk.text
                logger.info(f"Assistant: {chunk.text}")
            if hasattr(chunk, 'audio'):
                playback_queue.put(chunk.audio)

        # Check for exit command
        if any(cmd in response.lower() for cmd in EXIT_COMMANDS):
            break

        # Log performance
        cycle_time = time.time() - cycle_start
        total_cycles += 1
        total_response_time += cycle_time
        logger.info(f"Cycle completed in {cycle_time:.2f}s (avg: {total_response_time/total_cycles:.2f}s)")

    # Cleanup
    logger.info("Shutting down...")
    playback_queue.put(None)
    playback_thread.join()
    player.stop_stream()
    player.close()
    pyaudio.PyAudio().terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise