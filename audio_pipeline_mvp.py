#!/usr/bin/env python
"""
Audio-to-Text-to-Summary Pipeline MVP
Consolidated pipeline for processing audio files using Whisper and LM Studio
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

# Core libraries
import whisper
import openai
import requests
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa
import soundfile as sf

# Setup FFmpeg path for pydub
import subprocess
import warnings

# Enhanced logging setup
def setup_logging():
    """Setup comprehensive logging with multiple handlers"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logger
    logger = logging.getLogger("audio_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Detailed log file handler (DEBUG level)
    detailed_log_file = f"logs/audio_pipeline_detailed_{timestamp}.log"
    file_handler = logging.FileHandler(detailed_log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error log file handler (ERROR level only)
    error_log_file = f"logs/audio_pipeline_errors_{timestamp}.log"
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    # Performance log file handler (for timing information)
    perf_log_file = f"logs/audio_pipeline_performance_{timestamp}.log"
    perf_handler = logging.FileHandler(perf_log_file, encoding='utf-8')
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    perf_handler.setFormatter(perf_formatter)
    logger.addHandler(perf_handler)
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("AUDIO PIPELINE MVP STARTED")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Detailed logs: {detailed_log_file}")
    logger.info(f"Error logs: {error_log_file}")
    logger.info(f"Performance logs: {perf_log_file}")
    logger.info("=" * 60)
    
    return logger

# Initialize logging first
logger = setup_logging()

def setup_ffmpeg():
    """Setup FFmpeg paths for pydub with comprehensive detection"""
    logger.debug("Setting up FFmpeg paths...")
    
    # Common FFmpeg installation paths on Windows
    ffmpeg_paths = [
        "ffmpeg",  # If in PATH
        "C:/ffmpeg/ffmpeg.exe",
        "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
        "C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe",
        "C:/tools/ffmpeg/bin/ffmpeg.exe",
        "C:/chocolatey/bin/ffmpeg.exe",
        "C:/chocolatey/lib/ffmpeg/tools/ffmpeg.exe",
        "C:/Users/LOKYADAV/AppData/Local/Programs/ffmpeg/bin/ffmpeg.exe",
        "C:/Users/LOKYADAV/Desktop/ffmpeg/bin/ffmpeg.exe"
    ]
    
    ffmpeg_found = False
    ffmpeg_path = None
    
    for path in ffmpeg_paths:
        try:
            logger.debug(f"Testing FFmpeg path: {path}")
            result = subprocess.run([path, "-version"], 
                                  capture_output=True, 
                                  check=True, 
                                  timeout=10)
            if result.returncode == 0:
                ffmpeg_path = path
                ffmpeg_found = True
                logger.info(f"FFmpeg found at: {path}")
                break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug(f"FFmpeg not found at: {path}")
            continue
    
    if ffmpeg_found and ffmpeg_path:
        # Set pydub paths
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        
        # Set ffprobe path
        ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
        AudioSegment.ffprobe = ffprobe_path
        
        # Set environment variables for Whisper
        os.environ["FFMPEG_BINARY"] = ffmpeg_path
        os.environ["FFPROBE_BINARY"] = ffprobe_path
        
        # Add FFmpeg directory to PATH for Whisper
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        logger.info("FFmpeg configured successfully for pydub and Whisper")
        return True
    else:
        logger.warning("FFmpeg not found in common locations")
        logger.warning("Audio processing may fail. Please install FFmpeg:")
        logger.warning("  Option 1: choco install ffmpeg (run as Administrator)")
        logger.warning("  Option 2: Download from https://ffmpeg.org/download.html")
        logger.warning("  Option 3: Use alternative audio processing (limited functionality)")
        return False

# Setup FFmpeg
ffmpeg_configured = setup_ffmpeg()

# Suppress pydub warnings if FFmpeg is configured
if ffmpeg_configured:
    warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
else:
    # If FFmpeg is not available, try to use alternative methods
    logger.warning("FFmpeg not available - using fallback audio processing")
    try:
        # Try to use librosa as fallback for basic audio operations
        import librosa
        logger.info("Librosa available for fallback audio processing")
    except ImportError:
        logger.error("Neither FFmpeg nor librosa available for audio processing")
        logger.error("Please install FFmpeg or librosa for audio processing")

class AudioProcessor:
    """Handles audio preprocessing and chunking"""

    def __init__(self, chunk_length_ms: int = 30000):
        self.chunk_length_ms = chunk_length_ms
        logger.debug(f"AudioProcessor initialized with chunk_length_ms: {chunk_length_ms}")

    def preprocess_audio(self, audio_path: str, output_dir: str) -> List[str]:
        """Preprocess audio file: normalize, chunk if needed"""
        start_time = time.time()
        logger.info(f"Starting audio preprocessing for: {audio_path}")
        logger.debug(f"Output directory: {output_dir}")
        
        try:
            # Normalize paths for Windows compatibility
            audio_path = os.path.normpath(audio_path)
            output_dir = os.path.normpath(output_dir)
            
            logger.debug(f"Normalized audio path: {audio_path}")
            logger.debug(f"Normalized output directory: {output_dir}")
            
            # Ensure input file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.debug(f"Loading audio file: {audio_path}")
            audio = AudioSegment.from_file(audio_path)
            logger.debug(f"Audio loaded - Duration: {len(audio)}ms, Channels: {audio.channels}, Sample Rate: {audio.frame_rate}")

            # Normalize audio
            logger.debug("Normalizing audio...")
            audio = audio.normalize()
            logger.debug("Audio normalization completed")

            # Convert to mono if stereo
            if audio.channels > 1:
                logger.debug(f"Converting {audio.channels} channels to mono")
                audio = audio.set_channels(1)
                logger.debug("Stereo to mono conversion completed")

            # Set sample rate to 16kHz (optimal for Whisper)
            logger.debug(f"Converting sample rate from {audio.frame_rate} to 16000 Hz")
            audio = audio.set_frame_rate(16000)
            logger.debug("Sample rate conversion completed")

            # Create output directory with proper path handling
            logger.debug(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            # If audio is longer than 30 minutes, chunk it
            if len(audio) > 30 * 60 * 1000:  # 30 minutes in milliseconds
                logger.info(f"Audio is {len(audio)/1000/60:.1f} minutes long, chunking into {self.chunk_length_ms}ms segments")
                chunks = make_chunks(audio, self.chunk_length_ms)
                chunk_files = []

                for i, chunk in enumerate(chunks):
                    chunk_filename = os.path.normpath(os.path.join(output_dir, f"chunk_{i:03d}.wav"))
                    # Convert to forward slashes for compatibility
                    chunk_filename = chunk_filename.replace('\\', '/')
                    logger.debug(f"Exporting chunk {i+1}/{len(chunks)}: {chunk_filename}")
                    chunk.export(chunk_filename, format="wav")
                    chunk_files.append(chunk_filename)
                    logger.info(f"Created chunk {i+1}/{len(chunks)}: {chunk_filename}")

                processing_time = time.time() - start_time
                logger.info(f"Audio preprocessing completed in {processing_time:.2f} seconds")
                logger.debug(f"Generated {len(chunk_files)} chunk files")
                return chunk_files
            else:
                # Single file processing
                processed_file = os.path.normpath(os.path.join(output_dir, "processed_audio.wav"))
                # Convert to forward slashes for compatibility
                processed_file = processed_file.replace('\\', '/')
                logger.debug(f"Exporting single processed file: {processed_file}")
                audio.export(processed_file, format="wav")
                
                # Verify file was created
                if not os.path.exists(processed_file):
                    logger.error(f"Failed to create processed file: {processed_file}")
                    raise FileNotFoundError(f"Failed to create processed file: {processed_file}")
                
                processing_time = time.time() - start_time
                logger.info(f"Created processed file: {processed_file}")
                logger.info(f"Audio preprocessing completed in {processing_time:.2f} seconds")
                return [processed_file]

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error preprocessing audio after {processing_time:.2f} seconds: {e}")
            raise

class WhisperTranscriber:
    """Handles Whisper-based transcription"""

    def __init__(self, model_size: str = "large-v3", language: Optional[str] = None):
        self.model_size = model_size
        self.language = language
        self.model = None
        logger.debug(f"WhisperTranscriber initialized with model: {model_size}, language: {language}")
        self.load_model()

    def load_model(self):
        """Load Whisper model"""
        start_time = time.time()
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            logger.debug(f"Model loading started at: {datetime.now().isoformat()}")
            
            self.model = whisper.load_model(self.model_size)
            
            loading_time = time.time() - start_time
            logger.info(f"Whisper model loaded successfully in {loading_time:.2f} seconds")
            logger.debug(f"Model loading completed at: {datetime.now().isoformat()}")
            
        except Exception as e:
            loading_time = time.time() - start_time
            logger.error(f"Error loading Whisper model after {loading_time:.2f} seconds: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file using Whisper"""
        start_time = time.time()
        logger.info(f"Starting transcription for: {audio_path}")
        logger.debug(f"Transcription parameters - Language: {self.language}, Model: {self.model_size}")
        
        try:
            # Normalize path for Windows compatibility and convert to forward slashes
            audio_path = os.path.normpath(audio_path)
            # Convert backslashes to forward slashes for Whisper compatibility
            audio_path = audio_path.replace('\\', '/')
            logger.debug(f"Normalized audio path: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Get file size for logging
            file_size = os.path.getsize(audio_path)
            logger.debug(f"Audio file size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")

            # Configure transcription options
            options = {
                "language": self.language,
                "task": "transcribe",
                "verbose": True,
                "temperature": 0.0,
                "best_of": 5,
                "beam_size": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "suppress_tokens": [-1],
                "initial_prompt": None,
                "condition_on_previous_text": True,
                "fp16": False,  # Disable FP16 to avoid CPU warnings
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            logger.debug(f"Transcription options: {options}")

            # Transcribe
            logger.debug("Starting Whisper transcription...")
            transcription_start = time.time()
            
            try:
                result = self.model.transcribe(audio_path, **options)
            except FileNotFoundError as e:
                logger.warning(f"Whisper failed to load audio with FFmpeg: {e}")
                logger.info("Trying fallback method with librosa...")
                
                # Fallback: Load audio with librosa and pass numpy array to Whisper
                try:
                    import librosa
                    import numpy as np
                    
                    # Load audio with librosa
                    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                    logger.debug(f"Loaded audio with librosa: {len(audio_array)} samples, {sample_rate} Hz")
                    
                    # Transcribe with numpy array
                    result = self.model.transcribe(audio_array, **options)
                    logger.info("Successfully transcribed using librosa fallback")
                    
                except ImportError:
                    logger.error("Librosa not available for fallback")
                    raise
                except Exception as fallback_error:
                    logger.error(f"Fallback transcription failed: {fallback_error}")
                    raise
            
            transcription_time = time.time() - transcription_start
            
            # Add metadata
            result["audio_file"] = audio_path
            result["transcription_time"] = datetime.now().isoformat()
            result["model_used"] = self.model_size
            result["processing_time_seconds"] = transcription_time

            total_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds (total: {total_time:.2f}s)")
            logger.info(f"Text length: {len(result['text'])} characters")
            logger.debug(f"Transcription result keys: {list(result.keys())}")
            
            if 'segments' in result:
                logger.debug(f"Number of segments: {len(result['segments'])}")
                for i, segment in enumerate(result['segments'][:3]):  # Log first 3 segments
                    logger.debug(f"Segment {i}: {segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s")

            return result

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error transcribing audio after {total_time:.2f} seconds: {e}")
            raise

    def batch_transcribe(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files"""
        start_time = time.time()
        logger.info(f"Starting batch transcription of {len(audio_files)} files")
        
        results = []

        for i, audio_file in enumerate(audio_files):
            file_start_time = time.time()
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")
            logger.debug(f"File {i+1} processing started at: {datetime.now().isoformat()}")
            
            try:
                result = self.transcribe_audio(audio_file)
                results.append(result)
                
                file_time = time.time() - file_start_time
                logger.info(f"File {i+1} completed in {file_time:.2f} seconds")
                
            except Exception as e:
                file_time = time.time() - file_start_time
                logger.error(f"Failed to process file {i+1} after {file_time:.2f} seconds: {e}")
                # Add error result to maintain order
                results.append({
                    "error": str(e),
                    "audio_file": audio_file,
                    "processing_time_seconds": file_time
                })

        total_time = time.time() - start_time
        successful = len([r for r in results if "error" not in r])
        logger.info(f"Batch transcription completed in {total_time:.2f} seconds")
        logger.info(f"Successfully processed: {successful}/{len(audio_files)} files")
        
        return results

class LMStudioClient:
    """Client for interacting with LM Studio API"""

    def __init__(self, base_url: str = "http://localhost:1234", model_name: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        logger.debug(f"LMStudioClient initialized with base_url: {self.base_url}, model: {model_name}")
        
        self.client = openai.OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="lm-studio"  # LM Studio uses a dummy key
        )
        logger.debug("OpenAI client configured for LM Studio")

    def list_models(self) -> List[str]:
        """List available models"""
        start_time = time.time()
        try:
            logger.debug(f"Requesting models from: {self.base_url}/v1/models")
            response = requests.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            models = response.json()["data"]
            model_list = [model["id"] for model in models]
            
            request_time = time.time() - start_time
            logger.debug(f"Models request completed in {request_time:.2f} seconds")
            logger.debug(f"Available models: {model_list}")
            
            return model_list
        except Exception as e:
            request_time = time.time() - start_time
            logger.error(f"Error listing models after {request_time:.2f} seconds: {e}")
            return []

    def generate_summary(self, text: str, summary_type: str = "comprehensive") -> str:
        """Generate summary using LM Studio"""
        start_time = time.time()
        logger.info(f"Generating {summary_type} summary...")
        logger.debug(f"Summary generation started at: {datetime.now().isoformat()}")
        logger.debug(f"Input text length: {len(text)} characters")
        
        try:
            # Define different summary prompts
            prompts = {
                "comprehensive": f"""
Analyze the following text and provide a comprehensive summary with these sections:

1. **Main Topics**: Key themes and subjects discussed
2. **Key Insights**: Important findings, conclusions, or insights
3. **Action Items**: Specific actions, recommendations, or next steps mentioned
4. **Important Details**: Critical facts, figures, or specifics that shouldn't be missed
5. **Context**: Background information or situational context

Text to analyze:
{text}

Provide a well-structured summary:""",

                "actionable": f"""
Extract actionable insights from the following text. Focus on:

1. **Immediate Actions**: What needs to be done right away?
2. **Strategic Recommendations**: Long-term actions or strategies
3. **Key Decisions**: Important decisions that were made or need to be made
4. **Follow-ups**: Items that require follow-up or monitoring
5. **Resources Needed**: What resources, tools, or people are required?

Text to analyze:
{text}

Provide actionable insights:""",

                "bullet_points": f"""
Create a concise bullet-point summary of the following text:

• **Main Points**: Key topics in 3-5 bullet points
• **Action Items**: Specific actions in bullet format
• **Key Takeaways**: Important conclusions or insights
• **Next Steps**: What comes next

Text to analyze:
{text}

Provide bullet-point summary:"""
            }

            prompt = prompts.get(summary_type, prompts["comprehensive"])
            logger.debug(f"Using prompt for summary type: {summary_type}")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            # Get model name
            model_to_use = self.model_name or self.list_models()[0]
            logger.debug(f"Using model: {model_to_use}")

            # Generate summary
            api_start_time = time.time()
            logger.debug("Making API call to LM Studio...")
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a professional content analyst and summarizer. Provide clear, well-structured, and actionable summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                top_p=0.9
            )

            api_time = time.time() - api_start_time
            summary = response.choices[0].message.content
            
            total_time = time.time() - start_time
            logger.info(f"Summary generated successfully in {api_time:.2f} seconds (total: {total_time:.2f}s)")
            logger.info(f"Summary length: {len(summary)} characters")
            logger.debug(f"API response tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'unknown'}")
            
            return summary

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error generating {summary_type} summary after {total_time:.2f} seconds: {e}")
            raise

class AudioPipelineMVP:
    """Consolidated MVP pipeline orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing AudioPipelineMVP")
        logger.debug(f"Configuration: {config}")
        
        self.audio_processor = AudioProcessor(
            chunk_length_ms=config.get("chunk_length_ms", 30000)
        )
        self.transcriber = WhisperTranscriber(
            model_size=config.get("whisper_model", "large-v3"),
            language=config.get("language", None)
        )
        self.lm_client = LMStudioClient(
            base_url=config.get("lm_studio_url", "http://localhost:1234"),
            model_name=config.get("lm_studio_model", None)
        )
        logger.info("AudioPipelineMVP initialization completed")

    def process_single_file(self, audio_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single audio file"""
        pipeline_start_time = time.time()
        logger.info(f"Starting pipeline for: {audio_path}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Pipeline configuration: {self.config}")

        try:
            # Create output directories
            temp_dir = os.path.join(output_dir, "temp")
            logger.debug(f"Creating temp directory: {temp_dir}")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)  # Ensure temp directory exists

            # Step 1: Preprocess audio
            step1_start = time.time()
            logger.info("Step 1: Audio preprocessing")
            processed_files = self.audio_processor.preprocess_audio(audio_path, temp_dir)
            step1_time = time.time() - step1_start
            logger.info(f"Step 1 completed in {step1_time:.2f} seconds")

            # Verify processed files exist before transcription
            for file_path in processed_files:
                if not os.path.exists(file_path):
                    logger.error(f"Processed file not found: {file_path}")
                    raise FileNotFoundError(f"Processed file not found: {file_path}")
                else:
                    logger.debug(f"Verified processed file exists: {file_path}")

            # Step 2: Transcribe audio
            step2_start = time.time()
            logger.info("Step 2: Audio transcription")
            transcription_results = self.transcriber.batch_transcribe(processed_files)
            step2_time = time.time() - step2_start
            logger.info(f"Step 2 completed in {step2_time:.2f} seconds")

            # Step 3: Combine transcriptions if multiple chunks
            step3_start = time.time()
            logger.info("Step 3: Combining transcriptions")
            combined_text = self._combine_transcriptions(transcription_results)
            step3_time = time.time() - step3_start
            logger.info(f"Step 3 completed in {step3_time:.2f} seconds")
            logger.debug(f"Combined text length: {len(combined_text)} characters")

            # Step 4: Generate different types of summaries
            step4_start = time.time()
            logger.info("Step 4: Generating summaries")
            summaries = {}
            summary_types = ["comprehensive", "actionable", "bullet_points"]
            
            for i, summary_type in enumerate(summary_types):
                summary_start = time.time()
                logger.info(f"Generating {summary_type} summary ({i+1}/{len(summary_types)})")
                summaries[summary_type] = self.lm_client.generate_summary(
                    combined_text, summary_type
                )
                summary_time = time.time() - summary_start
                logger.info(f"{summary_type} summary completed in {summary_time:.2f} seconds")
            
            step4_time = time.time() - step4_start
            logger.info(f"Step 4 completed in {step4_time:.2f} seconds")

            # Step 5: Compile results
            step5_start = time.time()
            logger.info("Step 5: Compiling results")
            result = {
                "audio_file": audio_path,
                "processing_time": datetime.now().isoformat(),
                "transcription": {
                    "full_text": combined_text,
                    "segments": transcription_results,
                    "word_count": len(combined_text.split()),
                    "duration": sum(r.get("duration", 0) for r in transcription_results)
                },
                "summaries": summaries,
                "metadata": {
                    "whisper_model": self.config.get("whisper_model", "large-v3"),
                    "lm_studio_url": self.config.get("lm_studio_url", "http://localhost:1234"),
                    "chunks_processed": len(processed_files)
                },
                "performance": {
                    "step1_audio_preprocessing_seconds": step1_time,
                    "step2_transcription_seconds": step2_time,
                    "step3_combine_transcriptions_seconds": step3_time,
                    "step4_summarization_seconds": step4_time,
                    "step5_compile_results_seconds": time.time() - step5_start,
                    "total_pipeline_seconds": time.time() - pipeline_start_time
                }
            }
            step5_time = time.time() - step5_start
            logger.info(f"Step 5 completed in {step5_time:.2f} seconds")

            # Step 6: Save results
            step6_start = time.time()
            logger.info("Step 6: Saving results")
            self._save_results(result, output_dir)
            step6_time = time.time() - step6_start
            logger.info(f"Step 6 completed in {step6_time:.2f} seconds")

            # Clean up temporary files
            cleanup_start = time.time()
            logger.info("Cleaning up temporary files")
            self._cleanup_temp_files(temp_dir)
            cleanup_time = time.time() - cleanup_start
            logger.info(f"Cleanup completed in {cleanup_time:.2f} seconds")

            total_time = time.time() - pipeline_start_time
            logger.info("Pipeline completed successfully")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Performance breakdown:")
            logger.info(f"  - Audio preprocessing: {step1_time:.2f}s ({step1_time/total_time*100:.1f}%)")
            logger.info(f"  - Transcription: {step2_time:.2f}s ({step2_time/total_time*100:.1f}%)")
            logger.info(f"  - Text combination: {step3_time:.2f}s ({step3_time/total_time*100:.1f}%)")
            logger.info(f"  - Summarization: {step4_time:.2f}s ({step4_time/total_time*100:.1f}%)")
            logger.info(f"  - Results compilation: {step5_time:.2f}s ({step5_time/total_time*100:.1f}%)")
            logger.info(f"  - File saving: {step6_time:.2f}s ({step6_time/total_time*100:.1f}%)")
            logger.info(f"  - Cleanup: {cleanup_time:.2f}s ({cleanup_time/total_time*100:.1f}%)")
            
            return result

        except Exception as e:
            total_time = time.time() - pipeline_start_time
            logger.error(f"Pipeline error after {total_time:.2f} seconds: {e}")
            raise

    def _combine_transcriptions(self, transcription_results: List[Dict[str, Any]]) -> str:
        """Combine multiple transcription results into single text"""
        combined_text = ""
        for result in transcription_results:
            combined_text += result.get("text", "") + " "
        return combined_text.strip()

    def _save_results(self, result: Dict[str, Any], output_dir: str):
        """Save results to files"""
        save_start_time = time.time()
        logger.info(f"Saving results to: {output_dir}")
        
        try:
            # Save full results as JSON
            json_file = os.path.join(output_dir, "results.json")
            logger.debug(f"Saving JSON results to: {json_file}")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.debug(f"JSON results saved successfully")

            # Save transcription as text
            transcript_file = os.path.join(output_dir, "transcript.txt")
            logger.debug(f"Saving transcript to: {transcript_file}")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result["transcription"]["full_text"])
            logger.debug(f"Transcript saved successfully")

            # Save each summary type
            for summary_type, summary_content in result["summaries"].items():
                summary_file = os.path.join(output_dir, f"summary_{summary_type}.txt")
                logger.debug(f"Saving {summary_type} summary to: {summary_file}")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary_content)
                logger.debug(f"{summary_type} summary saved successfully")

            save_time = time.time() - save_start_time
            logger.info(f"Results saved successfully in {save_time:.2f} seconds")
            logger.debug(f"Files created in {output_dir}:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.debug(f"  - {file}: {file_size} bytes")

        except Exception as e:
            save_time = time.time() - save_start_time
            logger.error(f"Error saving results after {save_time:.2f} seconds: {e}")
            raise

    def _cleanup_temp_files(self, temp_dir: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(temp_dir):
                logger.debug(f"Cleaning up temp directory: {temp_dir}")
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.debug(f"Deleted temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {file_path}: {e}")
                
                # Try to remove the temp directory itself
                try:
                    os.rmdir(temp_dir)
                    logger.debug(f"Removed temp directory: {temp_dir}")
                except Exception as e:
                    logger.debug(f"Could not remove temp directory {temp_dir}: {e}")
            else:
                logger.debug(f"Temp directory does not exist: {temp_dir}")
                
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def find_audio_files(self, directory: str, recursive: bool = True) -> List[str]:
        """Find all audio files in directory"""
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.webm'}
        audio_files = []

        path = Path(directory)
        if not path.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        # Use glob patterns based on recursive flag
        patterns = ['**/*'] if recursive else ['*']

        for pattern in patterns:
            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    audio_files.append(str(file_path))

        logger.info(f"Found {len(audio_files)} audio files")
        return sorted(audio_files)

    def process_batch_parallel(self, audio_files: List[str], output_base_dir: str, max_workers: int = 4) -> Dict[str, Any]:
        """Process multiple audio files in parallel"""
        if not audio_files:
            logger.warning("No audio files to process")
            return {"results": [], "summary": {"total": 0, "success": 0, "failed": 0}}

        logger.info(f"Starting batch processing of {len(audio_files)} files")
        logger.info(f"Using {max_workers} parallel workers")

        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)

        # Prepare arguments for parallel processing
        args_list = [(audio_file, output_base_dir) for audio_file in audio_files]

        results = []

        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(self._process_single_file_worker, args): args for args in args_list}

            for future in concurrent.futures.as_completed(future_to_args):
                try:
                    result = future.result()
                    results.append(result)

                    # Log progress
                    completed = len(results)
                    total = len(audio_files)
                    success_count = len([r for r in results if r["status"] == "success"])
                    logger.info(f"Progress: {completed}/{total} files processed ({success_count} successful)")

                except Exception as e:
                    logger.error(f"Future exception: {e}")

        # Calculate summary statistics
        total_files = len(audio_files)
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "error"])

        summary = {
            "total": total_files,
            "success": successful, 
            "failed": failed,
            "success_rate": (successful / total_files * 100) if total_files > 0 else 0
        }

        # Save batch report
        batch_report = {
            "summary": summary,
            "results": results,
            "config": self.config
        }

        report_file = os.path.join(output_base_dir, "batch_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)

        logger.info(f"Batch processing completed:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"  Report saved: {report_file}")

        return batch_report

    def _process_single_file_worker(self, args) -> Dict[str, Any]:
        """Process a single audio file (for use with ThreadPoolExecutor)"""
        audio_file, output_base_dir = args

        try:
            # Create output directory for this file
            file_name = Path(audio_file).stem
            output_dir = os.path.join(output_base_dir, f"output_{file_name}")

            logger.info(f"Processing: {audio_file}")
            result = self.process_single_file(audio_file, output_dir)

            return {
                "status": "success",
                "audio_file": audio_file,
                "output_dir": output_dir,
                "result": result
            }

        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            return {
                "status": "error", 
                "audio_file": audio_file,
                "error": str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Audio-to-Text-to-Summary Pipeline MVP")
    parser.add_argument("input_path", help="Path to audio file or directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--language", help="Audio language (auto-detect if not specified)")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234", help="LM Studio server URL")
    parser.add_argument("--lm-studio-model", help="Specific LM Studio model to use")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search subdirectories recursively")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers for batch processing")

    args = parser.parse_args()

    # Configuration
    config = {
        "whisper_model": args.whisper_model,
        "language": args.language,
        "lm_studio_url": args.lm_studio_url,
        "lm_studio_model": args.lm_studio_model,
        "chunk_length_ms": 30000
    }

    # Initialize pipeline
    pipeline = AudioPipelineMVP(config)

    try:
        if os.path.isdir(args.input_path):
            # Batch processing
            audio_files = pipeline.find_audio_files(args.input_path, args.recursive)

            if not audio_files:
                logger.error("No audio files found")
                sys.exit(1)

            logger.info(f"Found {len(audio_files)} audio files for batch processing")
            result = pipeline.process_batch_parallel(audio_files, args.output_dir, args.max_workers)

            # Exit with error code if any files failed
            if result["summary"]["failed"] > 0:
                sys.exit(1)

        else:
            # Single file processing
            result = pipeline.process_single_file(args.input_path, args.output_dir)
            logger.info("Processing completed successfully!")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 