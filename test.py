import asyncio
import sounddevice as sd
import numpy as np
from mlx_lm import load, generate
import re
import logging
from cachetools import TTLCache, cached
from openai import OpenAI  # Ensure you have the OpenAI library installed
import pyaudio
import threading
import queue
import sys
import os
import webrtcvad  # For Voice Activity Detection
from collections import deque
import time  # For performance measurements
import platform

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import our custom wrappers
from whisper_mlx_wrapper import WhisperWrapper

# Check if we're on Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine().startswith('arm')

# Import kokoro-onnx wrapper for Apple Silicon acceleration if available
kokoro_onnx_tts = None
try:
    if IS_APPLE_SILICON:
        from kokoro_onnx_wrapper import KokoroOnnxWrapper
        logger.info("Apple Silicon detected, initializing kokoro-onnx for TTS acceleration")
        kokoro_onnx_tts = KokoroOnnxWrapper()
        if kokoro_onnx_tts.is_available():
            logger.info("kokoro-onnx initialized successfully for Apple Metal acceleration")
        else:
            logger.warning("kokoro-onnx initialization failed, will use standard TTS")
            kokoro_onnx_tts = None
    else:
        logger.info("Not running on Apple Silicon, will use standard TTS")
except ImportError:
    logger.info("kokoro-onnx wrapper not available, will use standard TTS")

# Preload models with error handling and optimization for real-time performance
try:
    # Use MLX acceleration on Apple Silicon if available
    logger.info("Initializing Whisper model with MLX acceleration if available")

    faster_whisper_model = WhisperWrapper(
        "tiny.en",  # Smallest model for fastest inference
        device="auto",  # Auto-select MPS if available
        compute_type="default"  # Use default precision (float16) for better compatibility
    )
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

try:
    # Create models directory if it doesn't exist
    os.makedirs("./models/mlx", exist_ok=True)

    # Set the HF_HOME environment variable to cache models
    os.environ["HF_HOME"] = os.path.abspath("./models")

    # Load the smallest MLX model for faster inference
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    logger.info("MLX-LM model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load MLX-LM model or tokenizer: {e}")
    raise

# Initialize cache for common responses
response_cache = TTLCache(maxsize=1000, ttl=300)  # Cache up to 1000 responses for 5 minutes

# Precompile regex patterns for performance (if needed)
QUOTE_PATTERN = re.compile(r"<quote>.*?</quote>", re.DOTALL)
PROBANTE_PATTERN = re.compile(r'\bprobante\b', re.IGNORECASE)
REPEAT_PATTERN = re.compile(r"(You're welcome\.)+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r'\s+')

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms (original value for stability)
VAD_MODE = 3  # Aggressiveness mode (0-3, 3 is most aggressive)

# Initialize OpenAI client for Kokoro TTS
try:
    client = OpenAI(
        base_url="http://localhost:8880/v1",  # Kokoro TTS is running on port 8880
        api_key="not-needed"
    )
    logger.info("OpenAI client for Kokoro TTS initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client for Kokoro TTS: {e}")
    raise

# Initialize PyAudio player
try:
    player = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
        frames_per_buffer=1024
    )
    logger.info("PyAudio player initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PyAudio player: {e}")
    raise

# Initialize Voice Activity Detector
vad = webrtcvad.Vad(VAD_MODE)

# Thread-safe queue for audio playback
playback_queue = queue.Queue()

# Conversation history for context maintenance
conversation_history = deque(maxlen=10)  # Store last 10 exchanges

# Exit command
EXIT_COMMANDS = {"goodbye", "exit", "quit", "stop"}

def recognize_speech_from_array(audio_array):
    """
    Transcribes audio from a NumPy array using preloaded Whisper model.
    Optimized for real-time performance with MLX acceleration on Apple Silicon.
    """
    if audio_array is None or len(audio_array) == 0:
        logger.error("Empty audio data provided for transcription.")
        return ""

    try:
        # Start timing for performance measurement
        start_time = time.time()

        # Transcribe using the wrapper (handles both MLX and faster-whisper)
        result = faster_whisper_model.transcribe(
            audio_array,
            language='en',
            beam_size=1,           # Reduce beam size for faster inference
            best_of=1,             # Only return the best result
            temperature=0.0,       # Deterministic output
            vad_filter=True,       # Filter out non-speech
            # These parameters are only used by faster-whisper, not MLX
            compression_ratio_threshold=2.4,  # More aggressive audio compression
            condition_on_previous_text=False, # Don't condition on previous text for faster processing
            vad_parameters=dict(min_silence_duration_ms=500)  # Shorter silence detection
        )

        # Extract text from the result
        if isinstance(result, dict) and "text" in result:
            # MLX or wrapper format
            transcription = result["text"]
        else:
            # Direct faster-whisper format (fallback)
            transcription = " ".join(segment.text for segment in result)

        # Log performance metrics
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f}s: {transcription}")

        return transcription.strip()
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return ""

@cached(cache=response_cache)
def query_mlx_lm_cached(transcription, max_tokens=300):
    """
    Generates a response using the preloaded MLX-LM model and tokenizer with caching.
    Optimized for real-time performance.
    """
    try:
        # Start timing for performance measurement
        start_time = time.time()

        # Append user message to history
        conversation_history.append({"role": "user", "content": transcription})

        # Prepare messages for context - limit to last 4 exchanges for faster processing
        messages = list(conversation_history)[-8:] if len(conversation_history) > 8 else list(conversation_history)

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        logger.info("Generating response from MLX-LM...")

        # Optimize generation parameters for speed
        # Note: MLX-LM's generate() has limited parameters
        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,  # Reduced max tokens for faster generation
        )

        # Log performance metrics
        elapsed = time.time() - start_time
        logger.info(f"MLX-LM response generated in {elapsed:.2f}s: {response_text[:50]}...")

        # Append assistant response to history
        conversation_history.append({"role": "assistant", "content": response_text})

        return response_text
    except Exception as e:
        logger.error(f"Error generating response with MLX-LM: {e}")
        return "I'm sorry, I couldn't process that. Could you please try again?"

def tts_stream_via_openai(text, voice="af_heart", speed=1.2):
    """
    Streams TTS audio directly to speakers using Kokoro TTS.
    Uses Apple Metal acceleration via kokoro-onnx on Apple Silicon if available,
    otherwise falls back to OpenAI-compatible API.
    Highly optimized for real-time performance.

    Args:
        text (str): The text to be converted to speech.
        voice (str): The voice to use for TTS.
        speed (float): Speed factor for speech (0.5-2.0).
    """
    # Start timing for performance measurement
    start_time = time.time()

    # Check if we can use kokoro-onnx for Apple Metal acceleration
    if kokoro_onnx_tts is not None and kokoro_onnx_tts.is_available():
        logger.info(f"Starting TTS via kokoro-onnx with Apple Metal acceleration...")
        try:
            # Split text into smaller chunks for faster parallel processing
            sentences = re.split(r'(?<=[.!?,:;]) +', text)

            # Process each sentence
            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Generate speech with kokoro-onnx
                audio_result = kokoro_onnx_tts.generate_speech(sentence, voice=voice, speed=speed)

                if audio_result is not None and audio_result[0] is not None:
                    # Extract audio and sample rate
                    audio, _ = audio_result  # We don't need the sample rate here

                    # Convert to int16 PCM format
                    audio_int16 = (audio * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()

                    # Send to playback queue
                    playback_queue.put(audio_bytes)
                else:
                    logger.warning(f"Failed to generate speech for sentence: {sentence}")

            # Log performance metrics
            elapsed = time.time() - start_time
            logger.info(f"TTS via kokoro-onnx completed in {elapsed:.2f}s.")
            return

        except Exception as e:
            logger.error(f"Error during kokoro-onnx TTS: {e}, falling back to API")
            # Fall back to API method

    # If kokoro-onnx is not available or failed, use the API method
    logger.info("Starting TTS streaming via Kokoro API...")
    try:
        # Split text into smaller chunks for faster parallel processing
        # Use more aggressive sentence splitting for faster initial response
        sentences = re.split(r'(?<=[.!?,:;]) +', text)

        # Process first sentence immediately for faster initial response
        if sentences:
            first_sentence = sentences[0]
            with client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice=voice,  # Use the specified voice
                response_format="pcm",
                input=first_sentence,
                speed=speed,  # Use the specified speed
                # Add optimization parameters
                normalization_options={
                    "normalize": False  # Skip normalization for speed
                }
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):  # Even larger chunks for efficiency
                    if chunk:
                        playback_queue.put(chunk)

            # Process remaining sentences in smaller batches for faster streaming
            if len(sentences) > 1:
                # Process in batches of 2-3 sentences for better streaming performance
                remaining_sentences = sentences[1:]
                batch_size = 3  # Process 3 sentences at a time

                for i in range(0, len(remaining_sentences), batch_size):
                    batch = " ".join(remaining_sentences[i:i+batch_size])
                    with client.audio.speech.with_streaming_response.create(
                        model="kokoro",
                        voice=voice,  # Use the specified voice
                        response_format="pcm",
                        input=batch,
                        speed=speed,  # Use the specified speed
                        normalization_options={
                            "normalize": False  # Skip normalization for speed
                        }
                    ) as response:
                        for chunk in response.iter_bytes(chunk_size=4096):
                            if chunk:
                                playback_queue.put(chunk)

        # Log performance metrics
        elapsed = time.time() - start_time
        logger.info(f"TTS streaming completed in {elapsed:.2f}s.")
    except Exception as e:
        logger.error(f"Error during TTS streaming: {e}")
        # Fallback to simpler TTS if the optimized approach fails
        try:
            logger.info("Attempting fallback TTS method...")
            with client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice="af_heart",
                response_format="pcm",
                input=text,
                speed=1.0  # Default speed for fallback
            ) as response:
                for chunk in response.iter_bytes(chunk_size=2048):
                    if chunk:
                        playback_queue.put(chunk)
            logger.info("Fallback TTS completed.")
        except Exception as fallback_error:
            logger.error(f"Fallback TTS also failed: {fallback_error}")

def playback_worker():
    """
    Worker thread to handle audio playback from the queue.
    """
    logger.info("Audio playback thread started.")
    while True:
        chunk = playback_queue.get()
        if chunk is None:
            logger.info("Audio playback thread received shutdown signal.")
            break
        try:
            player.write(chunk)
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
        finally:
            playback_queue.task_done()
    logger.info("Audio playback thread terminated.")

async def process_tts(text, voice="af_heart", speed=1.2):
    """
    Asynchronously processes the TTS streaming.

    Args:
        text (str): The text to be converted to speech.
        voice (str): The voice to use for TTS.
        speed (float): Speed factor for speech (0.5-2.0).
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: tts_stream_via_openai(text, voice, speed))

async def main():
    logger.info("Starting conversational voice assistant with real-time optimizations...")

    # Create models directory if it doesn't exist
    os.makedirs("./models/whisper", exist_ok=True)
    os.makedirs("./models/mlx", exist_ok=True)

    # Start playback worker thread
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    # Track performance metrics
    total_cycles = 0
    total_response_time = 0

    # Preload cache with common responses
    query_mlx_lm_cached("Hello")

    # Play startup sound to indicate readiness
    logger.info("Voice assistant ready. Say something...")

    while True:
        cycle_start_time = time.time()
        logger.info("Awaiting user input...")

        # Record audio with VAD - optimized for faster response
        audio_data = await record_audio_vad()

        if audio_data is None:
            logger.warning("No audio detected. Continuing...")
            continue

        # Start processing immediately after speech is detected
        processing_start = time.time()
        logger.info(f"Starting speech processing {processing_start - cycle_start_time:.2f}s after cycle start")

        # Start transcription task in parallel
        transcription_task = asyncio.create_task(asyncio.to_thread(recognize_speech_from_array, audio_data))

        # While transcription is running, we could do other preparation work here
        # This parallel processing reduces perceived latency

        # Wait for transcription to complete
        transcription = await transcription_task

        if not transcription:
            logger.warning("No transcription available. Continuing...")
            continue

        # Check for exit command
        if any(cmd in transcription.lower() for cmd in EXIT_COMMANDS):
            logger.info("Exit command received. Terminating conversation.")
            break

        # Start MLX-LM query task
        mlx_response_task = asyncio.create_task(asyncio.to_thread(query_mlx_lm_cached, transcription))

        # Wait for MLX-LM response
        mlx_response = await mlx_response_task

        if mlx_response.startswith("Error") or mlx_response.startswith("I'm sorry"):
            logger.error(f"MLX-LM Response Error: {mlx_response}")
            continue

        # Stream TTS response
        await process_tts(mlx_response)

        # Calculate and log cycle performance
        cycle_time = time.time() - cycle_start_time
        total_cycles += 1
        total_response_time += cycle_time
        avg_response_time = total_response_time / total_cycles

        logger.info(f"Cycle completed in {cycle_time:.2f}s (avg: {avg_response_time:.2f}s)")

    # Shutdown sequence
    logger.info("Shutting down conversational assistant...")
    playback_queue.put(None)  # Signal playback thread to exit
    playback_thread.join()
    shutdown()

async def record_audio_vad():
    """
    Records audio using Voice Activity Detection to determine when to start and stop recording.
    Optimized for real-time performance.

    Returns:
        np.ndarray: Recorded audio data as a NumPy array.
    """
    # Start timing for performance measurement
    start_time = time.time()

    logger.info("Recording started. Speak into the microphone...")
    audio_frames = []
    frame_length = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # For 30ms, 16000Hz: 16000 * 0.03 = 480 samples
    vad = webrtcvad.Vad(VAD_MODE)  # Set aggressiveness mode (0-3)

    # Optimize VAD parameters for faster response
    padding_duration_ms = 200  # Original value for stability
    num_padding_frames = padding_duration_ms // FRAME_DURATION
    ring_buffer = deque(maxlen=num_padding_frames)
    triggered = False

    # Add timeout to prevent hanging if no speech is detected
    max_silence_duration = 0.8  # seconds (original value for stability)
    silence_start_time = None

    # Add maximum recording duration to prevent overly long inputs
    max_recording_duration = 15.0  # seconds
    recording_start_time = None

    try:
        # Use standard stream parameters for stability
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                              blocksize=frame_length,
                              dtype='int16',
                              channels=CHANNELS,
                              latency='low') as stream:
            while True:
                try:
                    # Read raw audio data
                    data, overflow = stream.read(frame_length)

                    if overflow:
                        logger.warning("Audio overflow detected.")
                except Exception as e:
                    logger.warning(f"Error reading audio data: {e}")
                    continue  # Skip this frame and try again

                # Validate frame data before processing
                if len(data) != frame_length * 2:  # 2 bytes per int16 sample
                    logger.warning(f"Incorrect frame length: {len(data)} bytes. Expected {frame_length * 2} bytes.")
                    continue  # Skip this frame

                # Process the audio data regardless of its type
                try:
                    # The VAD library can handle both bytes and buffer objects
                    # No need to check the type, just ensure it's not None
                    if data is None:
                        logger.warning("Received None data from audio stream")
                        continue  # Skip this frame
                except Exception as e:
                    logger.error(f"Error validating frame data: {e}")
                    continue  # Skip this frame

                # Speech detection logic
                if not triggered:
                    # Not yet triggered, add to ring buffer
                    ring_buffer.append(data)

                    # Count voiced frames in buffer with error handling
                    try:
                        num_voiced = 0
                        valid_frames = 0
                        for f in ring_buffer:
                            try:
                                # Try to detect speech in this frame
                                is_speech = vad.is_speech(f, SAMPLE_RATE)
                                num_voiced += 1 if is_speech else 0
                                valid_frames += 1
                            except Exception as e:
                                # Just log and continue with other frames
                                logger.debug(f"Skipping frame in speech detection: {e}")

                        # Calculate speech ratio based on valid frames
                        if valid_frames > 0:
                            speech_ratio = num_voiced / valid_frames
                        else:
                            speech_ratio = 0
                    except Exception as e:
                        logger.error(f"Error calculating speech ratio: {e}")
                        speech_ratio = 0  # Default to no speech on error

                    # Trigger if enough speech detected
                    if speech_ratio > 0.5:  # Original threshold for reliable detection
                        triggered = True
                        recording_start_time = time.time()
                        audio_frames.extend(ring_buffer)
                        ring_buffer.clear()
                        logger.info("Speech detected. Recording...")
                else:
                    # Already triggered, collecting speech
                    audio_frames.append(data)
                    ring_buffer.append(data)

                    # Check for end of speech with error handling
                    try:
                        num_unvoiced = 0
                        valid_frames = 0
                        for f in ring_buffer:
                            try:
                                # Try to detect speech in this frame
                                is_speech = vad.is_speech(f, SAMPLE_RATE)
                                num_unvoiced += 0 if is_speech else 1
                                valid_frames += 1
                            except Exception as e:
                                # Just log and continue with other frames
                                logger.debug(f"Skipping frame in silence detection: {e}")

                        # Calculate silence ratio based on valid frames
                        if valid_frames > 0:
                            silence_ratio = num_unvoiced / valid_frames
                        else:
                            silence_ratio = 0
                    except Exception as e:
                        logger.error(f"Error calculating silence ratio: {e}")
                        silence_ratio = 0  # Default to no silence on error

                    # Set silence start time if we're starting to detect silence
                    if silence_ratio > 0.5 and silence_start_time is None:  # Original threshold for reliable silence detection
                        silence_start_time = time.time()
                    # Reset silence timer if speech detected again
                    elif silence_ratio <= 0.5 and silence_start_time is not None:
                        silence_start_time = None

                    # End recording if silence threshold reached
                    if silence_start_time and (time.time() - silence_start_time) > max_silence_duration:
                        logger.info(f"Speech ended after {max_silence_duration}s of silence.")
                        break

                    # End recording if max duration reached
                    if recording_start_time and (time.time() - recording_start_time) > max_recording_duration:
                        logger.info(f"Max recording duration of {max_recording_duration}s reached.")
                        break

        # Convert audio_frames to NumPy array
        audio_data = b''.join(audio_frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize

        # Log performance metrics
        elapsed = time.time() - start_time
        logger.info(f"Audio recording completed in {elapsed:.2f}s, {len(audio_np)/SAMPLE_RATE:.2f}s of audio")

        return audio_np
    except Exception as e:
        logger.error(f"Error during VAD recording: {e}")
        return None

def shutdown():
    logger.info("Shutting down PyAudio player...")
    try:
        player.stop_stream()
        player.close()
        pyaudio.PyAudio().terminate()
        logger.info("PyAudio player shut down successfully.")
    except Exception as e:
        logger.error(f"Error during PyAudio shutdown: {e}")

# Run the workflow with graceful shutdown
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Cleaning up...")
        shutdown()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        shutdown()
        sys.exit(1)