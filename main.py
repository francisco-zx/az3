"""
Audio Transcription and Diarization Project
==========================================

A Python project that combines OpenAI's Whisper for speech recognition
with pyannote.audio for speaker diarization to create timestamped
transcripts with speaker labels.

Requirements:
- openai-whisper
- pyannote.audio
- torch
- torchaudio
- librosa
- soundfile
"""

import os
import warnings
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import logging
from scipy.signal import butter, filtfilt, wiener
import time
import re
from typing import Any

import torch
import whisper
import numpy as np
from pyannote.audio import Pipeline, Inference
from pyannote.core import Annotation, Segment
# Prefer SpeechBrain encoder; fall back to pyannote wrapper if needed
try:
    from speechbrain.inference.speaker import EncoderClassifier
    _HAS_SPEECHBRAIN = True
except Exception:
    _HAS_SPEECHBRAIN = False
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import librosa
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# Optional OpenAI SDK for AI-based summarization
try:
    from openai import OpenAI  # New SDK (2023+)
    _HAS_OPENAI_SDK = True
    _HAS_OPENAI_LEGACY = False
except Exception:
    try:
        import openai as openai_legacy  # Legacy SDK
        _HAS_OPENAI_SDK = False
        _HAS_OPENAI_LEGACY = True
    except Exception:
        _HAS_OPENAI_SDK = False
        _HAS_OPENAI_LEGACY = False

warnings.filterwarnings("ignore", category=UserWarning)

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Represents a persistent speaker profile with embeddings and metadata."""
    speaker_id: str
    name: Optional[str] = None
    embeddings: List[np.ndarray] = None
    files_appeared: List[str] = None
    total_speaking_time: float = 0.0
    segment_count: int = 0
    created_at: str = ""
    last_seen: str = ""
    
    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = []
        if self.files_appeared is None:
            self.files_appeared = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def add_embedding(self, embedding: np.ndarray, file_path: str = "", duration: float = 0.0):
        """Add a new embedding to this speaker's profile."""
        # Store L2-normalized embedding for consistency
        if embedding is not None:
            norm = np.linalg.norm(embedding) or 1.0
            self.embeddings.append((embedding / norm).astype(np.float32))
        if file_path and file_path not in self.files_appeared:
            self.files_appeared.append(file_path)
        self.total_speaking_time += duration
        self.segment_count += 1
        self.last_seen = datetime.now().isoformat()
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get the average embedding for this speaker."""
        if not self.embeddings:
            return None
        avg = np.mean(self.embeddings, axis=0)
        norm = np.linalg.norm(avg) or 1.0
        return (avg / norm).astype(np.float32)
    
    def similarity_to_embedding(self, embedding: np.ndarray, threshold: float = 0.7) -> float:
        """Calculate similarity to a given embedding."""
        if not self.embeddings:
            return 0.0
        
        avg_embedding = self.get_average_embedding()
        if avg_embedding is None:
            return 0.0
        
        # Ensure inputs are 1D vectors
        a = np.asarray(embedding).reshape(-1)
        b = np.asarray(avg_embedding).reshape(-1)
        similarity = cosine_similarity([a], [b])[0][0]
        return max(0.0, similarity)  # Ensure non-negative


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio with speaker and timing info."""
    start_time: float
    end_time: float
    speaker: str
    text: str
    confidence: Optional[float] = None
    speaker_confidence: Optional[float] = None  # Confidence in speaker identification
    
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_srt_format(self, index: int) -> str:
        """Convert to SRT subtitle format."""
        start = self._format_timestamp(self.start_time)
        end = self._format_timestamp(self.end_time)
        return f"{index}\n{start} --> {end}\n[{self.speaker}] {self.text}\n"

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"


class AudioProcessor:
    """Handles audio file loading and preprocessing."""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return as tensor with sample rate."""
        try:
            # Load with librosa for consistent preprocessing
            audio, sr = librosa.load(file_path, sr=self.target_sample_rate)
            audio_tensor = torch.from_numpy(audio).float()
            
            logger.info(f"Loaded audio: {audio_tensor.shape[0] / sr:.2f}s at {sr}Hz")
            return audio_tensor, sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")
    
    def preprocess_for_whisper(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocess audio for Whisper model."""
        # Ensure mono channel
        if audio.dim() > 1:
            audio = torch.mean(audio, dim=0)
        
        # Normalize amplitude
        if torch.max(torch.abs(audio)) > 0:
            audio = audio / torch.max(torch.abs(audio))
        
        return audio

    def enhance_audio(self, audio: torch.Tensor, sample_rate: int, 
                      bandpass_low_hz: int = 100, bandpass_high_hz: int = 8000,
                      use_denoise: bool = True) -> np.ndarray:
        """Apply simple enhancement: pre-emphasis, band-pass, and Wiener denoise.
        Returns enhanced mono waveform as float32 numpy array in [-1, 1].
        """
        logger.info(
            f"Enhancing audio: bandpass={bandpass_low_hz}-{bandpass_high_hz}Hz, denoise={use_denoise}"
        )
        # Convert to mono numpy
        if audio.dim() > 1:
            audio = torch.mean(audio, dim=0)
        y = audio.detach().cpu().numpy().astype(np.float32)

        # Pre-emphasis to boost higher frequencies
        try:
            y = librosa.effects.preemphasis(y, coef=0.97)
        except Exception:
            pass

        # Band-pass filter (Butterworth)
        try:
            nyq = 0.5 * sample_rate
            low = max(1.0, float(bandpass_low_hz)) / nyq
            high = min(float(bandpass_high_hz), nyq - 1.0) / nyq
            if 0 < low < high < 1:
                b, a = butter(4, [low, high], btype='band')
                y = filtfilt(b, a, y).astype(np.float32)
        except Exception:
            pass

        # Wiener denoise as a simple noise reduction (optional)
        if use_denoise:
            try:
                y = wiener(y).astype(np.float32)
            except Exception:
                pass

        # Normalize to [-1, 1]
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak > 0:
            y = (y / peak).astype(np.float32)
        y = np.clip(y, -1.0, 1.0)
        return y

    def save_temp_wav(self, audio: np.ndarray, sample_rate: int, prefix: str = "enhanced_") -> Path:
        """Save a temporary wav file and return its path."""
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        filename = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        temp_path = temp_dir / filename
        sf.write(temp_path, audio, sample_rate)
        return temp_path


class SpeakerProfileManager:
    """Manages persistent speaker profiles across different audio files."""
    
    def __init__(self, profiles_dir: str = "speaker_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.profiles_file = self.profiles_dir / "speaker_profiles.pkl"
        self.metadata_file = self.profiles_dir / "profiles_metadata.json"
        
        self.profiles: Dict[str, SpeakerProfile] = {}
        self.embedding_model = None
        self._load_profiles()
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the speaker embedding model."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if _HAS_SPEECHBRAIN:
                self.embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": str(device)}
                )
                logger.info("Loaded SpeechBrain EncoderClassifier for speaker embeddings")
            else:
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=device
                )
                logger.info("Loaded pyannote PretrainedSpeakerEmbedding for speaker embeddings")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {str(e)}")
            self.embedding_model = None
    
    def _load_profiles(self):
        """Load existing speaker profiles from disk."""
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, 'rb') as f:
                    self.profiles = pickle.load(f)
                logger.info(f"Loaded {len(self.profiles)} existing speaker profiles")
            except Exception as e:
                logger.warning(f"Could not load existing profiles: {str(e)}")
                self.profiles = {}
        else:
            self.profiles = {}
    
    def _save_profiles(self):
        """Save speaker profiles to disk."""
        try:
            # Save binary profiles
            with open(self.profiles_file, 'wb') as f:
                pickle.dump(self.profiles, f)
            
            # Save human-readable metadata
            metadata = {
                "total_profiles": len(self.profiles),
                "last_updated": datetime.now().isoformat(),
                "speakers": {
                    speaker_id: {
                        "name": profile.name,
                        "files_count": len(profile.files_appeared),
                        "total_speaking_time": profile.total_speaking_time,
                        "segment_count": profile.segment_count,
                        "created_at": profile.created_at,
                        "last_seen": profile.last_seen
                    }
                    for speaker_id, profile in self.profiles.items()
                }
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save profiles: {str(e)}")
    
    def export_signatures_json(self, output_path: str = None):
        """Export per-speaker signature (average embedding) to a JSON file.
        
        The signature is the L2-normalized average of all stored embeddings
        for each speaker profile. Useful for external matching or inspection.
        """
        try:
            if output_path is None:
                output_path = str(self.profiles_dir / "signatures.json")
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            export = {
                "generated_at": datetime.now().isoformat(),
                "total_profiles": len(self.profiles),
                "profiles": {}
            }
            for speaker_id, profile in self.profiles.items():
                avg = profile.get_average_embedding()
                if avg is not None:
                    # L2 normalize signature for consistency
                    norm = np.linalg.norm(avg) if np.linalg.norm(avg) > 0 else 1.0
                    signature = (avg / norm).tolist()
                else:
                    signature = None
                export["profiles"][speaker_id] = {
                    "name": profile.name,
                    "files_appeared": profile.files_appeared,
                    "total_speaking_time": profile.total_speaking_time,
                    "segment_count": profile.segment_count,
                    "created_at": profile.created_at,
                    "last_seen": profile.last_seen,
                    "signature": signature
                }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2)
            logger.info(f"Speaker signatures exported to {output_path}")
        except Exception as e:
            logger.warning(f"Could not export signatures: {str(e)}")
    
    def extract_speaker_embedding(self, audio_path: str, start_time: float, 
                                 end_time: float) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio segment."""
        if self.embedding_model is None:
            return None
        
        try:
            # Load audio and slice the desired segment
            audio, sr = librosa.load(audio_path, sr=16000)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            if segment_audio.size == 0:
                return None
            
            # Prepare waveform tensor
            waveform = np.asarray(segment_audio, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)
            # Skip segments that are too short (< 0.25s)
            if waveform.size < int(0.25 * sr):
                return None
            waveform_t = torch.from_numpy(waveform).float()
            with torch.no_grad():
                if _HAS_SPEECHBRAIN and hasattr(self.embedding_model, "encode_batch"):
                    # SpeechBrain expects [batch, time]
                    embedding = self.embedding_model.encode_batch(waveform_t.unsqueeze(0))
                else:
                    # pyannote wrapper expects [batch, channels, time]
                    embedding = self.embedding_model(waveform_t.unsqueeze(0).unsqueeze(0))

            # Handle possible tuple/list returns
            if isinstance(embedding, (tuple, list)) and len(embedding) > 0:
                embedding = embedding[0]

            # Convert to numpy regardless of backend tensor type
            if hasattr(embedding, "detach"):
                embedding_np = embedding.detach().cpu().numpy()
            else:
                embedding_np = np.asarray(embedding)
            
            return np.squeeze(embedding_np)
                    
        except Exception as e:
            logger.warning(f"Could not extract embedding: {str(e)}")
            return None
    
    def identify_speaker(self, embedding: np.ndarray, 
                        similarity_threshold: float = 0.75) -> Tuple[Optional[str], float]:
        """Identify speaker based on embedding similarity."""
        if embedding is None or len(self.profiles) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, profile in self.profiles.items():
            similarity = profile.similarity_to_embedding(embedding, similarity_threshold)
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = speaker_id
        
        return best_match, best_similarity
    
    def create_new_speaker(self, embedding: np.ndarray, file_path: str, 
                          duration: float, temp_speaker_id: str) -> str:
        """Create a new speaker profile."""
        # Generate a consistent speaker ID
        speaker_id = f"Speaker_{len(self.profiles) + 1:03d}"
        
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            embeddings=[embedding],
            files_appeared=[file_path],
            total_speaking_time=duration,
            segment_count=1
        )
        
        self.profiles[speaker_id] = profile
        logger.info(f"Created new speaker profile: {speaker_id}")
        return speaker_id
    
    def update_speaker_profile(self, speaker_id: str, embedding: np.ndarray, 
                             file_path: str, duration: float):
        """Update an existing speaker profile with new data."""
        if speaker_id in self.profiles:
            self.profiles[speaker_id].add_embedding(embedding, file_path, duration)
    
    def process_diarization_with_profiles(self, diarization: Annotation, 
                                        audio_path: str,
                                        similarity_threshold: float = 0.72,
                                        min_segments_per_speaker: int = 1) -> Dict[str, str]:
        """Process diarization results and map to persistent speaker IDs."""
        temp_to_persistent = {}  # Maps temporary speaker IDs to persistent ones
        embeddings_cache = {}    # Cache aggregated embeddings per temp speaker
        
        logger.info("Matching speakers to existing profiles...")
        
        # Collect multiple embeddings per temp speaker and average them
        temp_to_embeds: Dict[str, List[np.ndarray]] = {}
        temp_to_counts: Dict[str, int] = {}
        for segment, _, temp_speaker in diarization.itertracks(yield_label=True):
            temp_to_counts[temp_speaker] = temp_to_counts.get(temp_speaker, 0) + 1
            embedding = self.extract_speaker_embedding(audio_path, segment.start, segment.end)
            if embedding is not None:
                norm = np.linalg.norm(embedding) or 1.0
                embedding = (embedding / norm).astype(np.float32)
                temp_to_embeds.setdefault(temp_speaker, []).append(embedding)
        for temp_speaker, emb_list in temp_to_embeds.items():
            if len(emb_list) == 0:
                continue
            avg = np.mean(emb_list, axis=0)
            norm = np.linalg.norm(avg) or 1.0
            embeddings_cache[temp_speaker] = (avg / norm).astype(np.float32)
        
        # Match temporary speakers to persistent profiles
        for temp_speaker, embedding in embeddings_cache.items():
            if temp_speaker not in temp_to_persistent:
                # Try to identify existing speaker
                persistent_id, confidence = self.identify_speaker(embedding, similarity_threshold=similarity_threshold)
                
                if persistent_id:
                    temp_to_persistent[temp_speaker] = persistent_id
                    logger.info(f"Matched {temp_speaker} to existing {persistent_id} (similarity: {confidence:.3f})")
                else:
                    # Create new speaker profile
                    # Calculate total duration for this speaker
                    total_duration = sum(
                        segment.end - segment.start 
                        for segment, _, speaker in diarization.itertracks(yield_label=True)
                        if speaker == temp_speaker
                    )
                    
                    persistent_id = self.create_new_speaker(
                        embedding, audio_path, total_duration, temp_speaker
                    )
                    temp_to_persistent[temp_speaker] = persistent_id
        
        # Update profiles with new data
        for segment, _, temp_speaker in diarization.itertracks(yield_label=True):
            if temp_speaker in temp_to_persistent and temp_speaker in embeddings_cache:
                persistent_id = temp_to_persistent[temp_speaker]
                embedding = embeddings_cache[temp_speaker]
                duration = segment.end - segment.start
                
                self.update_speaker_profile(persistent_id, embedding, audio_path, duration)
        
        # Save updated profiles
        self._save_profiles()
        
        return temp_to_persistent
    
    def get_speaker_info(self, speaker_id: str) -> Optional[Dict]:
        """Get information about a specific speaker."""
        if speaker_id not in self.profiles:
            return None
        
        profile = self.profiles[speaker_id]
        return {
            "speaker_id": speaker_id,
            "name": profile.name,
            "files_appeared": profile.files_appeared,
            "total_speaking_time": profile.total_speaking_time,
            "segment_count": profile.segment_count,
            "created_at": profile.created_at,
            "last_seen": profile.last_seen
        }
    
    def list_all_speakers(self) -> List[Dict]:
        """Get information about all known speakers."""
        return [self.get_speaker_info(speaker_id) for speaker_id in self.profiles.keys()]
    
    def assign_speaker_name(self, speaker_id: str, name: str):
        """Assign a human-readable name to a speaker."""
        if speaker_id in self.profiles:
            self.profiles[speaker_id].name = name
            self._save_profiles()
            logger.info(f"Assigned name '{name}' to {speaker_id}")
        else:
            logger.warning(f"Speaker {speaker_id} not found")
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str, keep_name: str = None):
        """Merge two speaker profiles (useful for corrections)."""
        if speaker_id1 not in self.profiles or speaker_id2 not in self.profiles:
            print("One or both speakers not found")
            return
        
        profile1 = self.profiles[speaker_id1]
        profile2 = self.profiles[speaker_id2]
        
        # Merge data into profile1
        profile1.embeddings.extend(profile2.embeddings)
        profile1.files_appeared.extend(profile2.files_appeared)
        profile1.files_appeared = list(set(profile1.files_appeared))  # Remove duplicates
        profile1.total_speaking_time += profile2.total_speaking_time
        profile1.segment_count += profile2.segment_count
        
        if keep_name:
            profile1.name = keep_name
        elif profile2.name and not profile1.name:
            profile1.name = profile2.name
        
        # Remove profile2
        del self.profiles[speaker_id2]
        self._save_profiles()
        
        logger.info(f"Merged {speaker_id2} into {speaker_id1}")


class SpeakerDiarizer:
    """Handles speaker diarization using pyannote.audio."""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1"):
        self.model_name = model_name
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the diarization pipeline."""
        try:
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
            if hf_token:
                logger.info("Loading diarization pipeline with Hugging Face token from environment")
                self.pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=hf_token)
            else:
                logger.info("Loading diarization pipeline without token; set HUGGINGFACE_HUB_TOKEN or HF_TOKEN if access is gated")
                self.pipeline = Pipeline.from_pretrained(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("Using GPU for diarization")
            else:
                logger.info("Using CPU for diarization")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization model: {str(e)}")
    
    def diarize(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 10) -> Annotation:
        """Perform speaker diarization on audio file."""
        try:
            diarization = self.pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            speaker_count = len(diarization.labels())
            logger.info(f"Detected {speaker_count} unique speakers")
            
            return diarization
            
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {str(e)}")


class WhisperTranscriber:
    """Handles speech-to-text transcription using Whisper."""
    
    def __init__(self, model_size: str = "base", language: Optional[str] = None,
                 boost_accuracy: bool = False, segment_padding: float = 0.5):
        self.model_size = model_size
        self.language = language
        self.model = None
        self.boost_accuracy = boost_accuracy
        self.segment_padding = max(0.0, float(segment_padding))
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
    
    def detect_and_set_language(self, audio: torch.Tensor, sample_rate: int, detection_window_seconds: int = 60) -> Optional[str]:
        """Detect language once from the first N seconds and lock it for stability."""
        try:
            # Take first window for more robust detection
            end_sample = min(len(audio), detection_window_seconds * sample_rate)
            snippet = audio[:end_sample].numpy()
            options = {
                "language": None,  # force auto-detect
                "task": "transcribe",
                "fp16": torch.cuda.is_available()
            }
            result = self.model.transcribe(snippet, **options)
            lang = result.get("language")
            if lang:
                self.language = lang
                logger.info(f"Detected and fixed language: {lang}")
            else:
                logger.warning("Language detection returned no language; keeping previous setting")
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return None

    def transcribe_segment(self, audio: torch.Tensor, start_time: float, 
                          end_time: float, sample_rate: int) -> Dict:
        """Transcribe a specific audio segment."""
        try:
            # Extract segment with optional padding for context
            total_samples = audio.shape[0]
            pad_samples = int(self.segment_padding * sample_rate)
            start_sample = max(0, int(start_time * sample_rate) - pad_samples)
            end_sample = min(total_samples, int(end_time * sample_rate) + pad_samples)
            segment_audio = audio[start_sample:end_sample]
            
            # Convert to numpy for Whisper
            segment_numpy = segment_audio.numpy()
            
            # Transcribe with Whisper
            options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": torch.cuda.is_available(),
            }
            if self.boost_accuracy:
                options["temperature"] = 0.0
                options["condition_on_previous_text"] = False
                if self.language:
                    # Provide a light prompt to anchor language
                    options["initial_prompt"] = f"This audio is in {self.language}."
            
            result = self.model.transcribe(segment_numpy, **options)
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for segment {start_time:.2f}-{end_time:.2f}s: {str(e)}")
            return {"text": "[Transcription Error]"}


class AudioTranscriberDiarizer:
    """Main class that combines transcription and diarization."""
    
    def __init__(self, whisper_model: str = "base", language: Optional[str] = None,
                 diarization_model: str = "pyannote/speaker-diarization-3.1",
                 boost_accuracy: bool = False, segment_padding: float = 0.5,
                 enhancement_level: str = "off",
                 similarity_threshold: float = 0.72):
        self.audio_processor = AudioProcessor()
        self.diarizer = SpeakerDiarizer(diarization_model)
        self.transcriber = WhisperTranscriber(whisper_model, language,
                                              boost_accuracy=boost_accuracy,
                                              segment_padding=segment_padding)
        self.enhancement_level = enhancement_level
        self.similarity_threshold = similarity_threshold
        self.metrics: Dict[str, float] = {}
        self.profile_manager = SpeakerProfileManager()
        
    def process_audio(self, audio_path: str, min_speakers: int = 1, 
                     max_speakers: int = 10) -> List[TranscriptSegment]:
        """Process audio file for transcription and diarization."""
        logger.info(f"Processing audio file: {audio_path}")
        metrics: Dict[str, float] = {}
        t0_all = time.time()
        
        # Load and preprocess audio
        t0 = time.time()
        audio_tensor, sample_rate = self.audio_processor.load_audio(audio_path)
        metrics["load_audio_s"] = time.time() - t0
        
        # Optional enhancement to stabilize recognition
        if self.enhancement_level and self.enhancement_level != "off":
            start_enh = time.time()
            try:
                use_denoise = (self.enhancement_level == "medium")
                enhanced_np = self.audio_processor.enhance_audio(
                    audio_tensor, sample_rate, use_denoise=use_denoise
                )
                audio_tensor = torch.from_numpy(enhanced_np).float()
                logger.info(f"Applied audio enhancement level={self.enhancement_level}")
            except Exception as e:
                logger.warning(f"Audio enhancement failed, proceeding with raw audio: {str(e)}")
            finally:
                enh_dur = time.time() - start_enh
                metrics["enhance_s"] = enh_dur
                logger.info(f"Enhancement took {enh_dur:.2f}s")
        else:
            metrics["enhance_s"] = 0.0
        
        # Detect language once and lock it in transcriber (if not provided)
        if self.transcriber.language is None:
            t0 = time.time()
            self.transcriber.detect_and_set_language(audio_tensor, sample_rate)
            metrics["language_detect_s"] = time.time() - t0
        else:
            metrics["language_detect_s"] = 0.0
        
        # Perform speaker diarization
        logger.info("Performing speaker diarization...")
        t0 = time.time()
        diarization = self.diarizer.diarize(audio_path, min_speakers, max_speakers)
        metrics["diarization_s"] = time.time() - t0
        
        # Map temporary diarization labels to persistent speaker IDs and update profiles
        try:
            t0 = time.time()
            temp_to_persistent = self.profile_manager.process_diarization_with_profiles(
                diarization, audio_path,
                similarity_threshold=getattr(self, "similarity_threshold", 0.72)
            )
            metrics["profiles_map_s"] = time.time() - t0
        except Exception as e:
            logger.warning(f"Speaker profile processing failed: {str(e)}")
            temp_to_persistent = {}
            metrics["profiles_map_s"] = 0.0
        
        # Process each speaker segment
        logger.info("Transcribing speaker segments...")
        segments = []
        t0 = time.time()
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            mapped_speaker = temp_to_persistent.get(speaker, speaker)
            
            # Transcribe this segment
            result = self.transcriber.transcribe_segment(
                audio_tensor, start_time, end_time, sample_rate
            )
            
            # Extract text and confidence
            text = result.get("text", "").strip()
            
            if text and text != "[Transcription Error]":
                transcript_segment = TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    speaker=mapped_speaker,
                    text=text,
                    confidence=None  # Whisper doesn't provide segment-level confidence
                )
                segments.append(transcript_segment)
                
                logger.debug(f"[{mapped_speaker}] {start_time:.1f}s-{end_time:.1f}s: {text[:50]}...")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)
        metrics["transcription_s"] = time.time() - t0
        metrics["segments"] = float(len(segments))
        try:
            metrics["speakers_detected"] = float(len(diarization.labels()))
        except Exception:
            metrics["speakers_detected"] = 0.0
        metrics["process_audio_s"] = time.time() - t0_all
        self.metrics = metrics
        logger.info(f"Processing complete. Generated {len(segments)} transcript segments.")
        
        return segments
    
    def save_signatures(self, output_path: str = None):
        """Save speaker signatures to a JSON file via the profile manager."""
        self.profile_manager.export_signatures_json(output_path)
    
    def save_results(self, segments: List[TranscriptSegment], output_path: str, 
                    format: str = "json"):
        """Save transcription results in specified format."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        if format.lower() == "json":
            self._save_json(segments, output_path)
        elif format.lower() == "srt":
            self._save_srt(segments, output_path)
        elif format.lower() == "txt":
            self._save_txt(segments, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def summarize_conversation(self, segments: List[TranscriptSegment], max_chars: int = 1500) -> str:
        """Generate a richer, case-adaptive summary with participants, traits, key points, dates & places."""
        if not segments:
            return "No content to summarize."

        # Helpers
        def split_sentences(text: str) -> List[str]:
            parts = []
            for p in re.split(r"(?<=[\.\!\?…])\s+|\n+", text):
                p = p.strip()
                if len(p) > 0:
                    parts.append(p)
            return parts

        # Minimal multilingual stopwords (en/es)
        stop_en = {
            'the','a','an','and','or','but','if','then','so','because','to','of','in','on','for','with','at','by','from','as','is','are','was','were','be','been','it','this','that','these','those','we','you','they','i','he','she','them','us','our','your','their','my','me','do','does','did','will','would','can','could','should','have','has','had'
        }
        stop_es = {
            'el','la','los','las','un','una','unos','unas','y','o','pero','si','entonces','porque','para','de','del','en','con','a','por','como','es','son','fue','eran','ser','ha','han','está','están','esto','eso','estos','esas','nosotros','ustedes','ellos','yo','él','ella','nos','nuestro','su','sus','mi','me','hacer','hace','hizo','hará','puede','podría','debería','tener','tiene','tenía'
        }
        def is_stop(w: str) -> bool:
            wl = w.lower()
            return wl in stop_en or wl in stop_es or len(wl) <= 2

        # Basic stats per speaker
        speaker_stats: Dict[str, Dict[str, float]] = {}
        sentences: List[Tuple[str, str]] = []  # (speaker, sentence)
        for seg in segments:
            stats = speaker_stats.setdefault(seg.speaker, {"time": 0.0, "segments": 0, "words": 0})
            stats["time"] += seg.duration()
            stats["segments"] += 1
            stats["words"] += len(seg.text.split())
            for sent in split_sentences(seg.text):
                sentences.append((seg.speaker, sent))

        # Traits heuristics
        total_time = sum(st["time"] for st in speaker_stats.values()) or 1.0
        traits: Dict[str, List[str]] = {}
        for spk, st in speaker_stats.items():
            spk_traits = []
            share = st["time"] / total_time
            wpm = (st["words"] / (st["time"] / 60)) if st["time"] > 0 else 0
            avg_len = (st["time"] / st["segments"]) if st["segments"] > 0 else 0
            if share >= 0.5:
                spk_traits.append("dominant")
            if wpm >= 160:
                spk_traits.append("fast-paced")
            elif wpm > 0 and wpm < 110:
                spk_traits.append("measured")
            if avg_len >= 8:
                spk_traits.append("elaborate")
            elif avg_len > 0 and avg_len < 4:
                spk_traits.append("concise")
            # naive sentiment tone
            pos_words = {'great','good','excelente','bueno','genial','positivo','acuerdo','acordamos','agree','decide','decided'}
            neg_words = {'malo','mal','problema','issue','no','nunca','nobody','nadie','fail','falla','error'}
            tone_score = 0
            # sample a few sentences by speaker
            spk_sents = [s for sspk,s in sentences if sspk == spk][:20]
            for s in spk_sents:
                wl = s.lower()
                tone_score += sum(1 for w in pos_words if w in wl)
                tone_score -= sum(1 for w in neg_words if w in wl)
            if tone_score >= 2:
                spk_traits.append("positive")
            elif tone_score <= -2:
                spk_traits.append("critical")
            traits[spk] = spk_traits

        # Keywords & key sentences
        def tokenize(text: str) -> List[str]:
            return re.findall(r"[\wáéíóúñÁÉÍÓÚÑ]+", text.lower())
        freq: Dict[str,int] = {}
        for _, s in sentences:
            for w in tokenize(s):
                if not is_stop(w):
                    freq[w] = freq.get(w, 0) + 1
        top_keywords = [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:20]]

        # Score sentences by keyword coverage and uniqueness
        def sent_score(s: str) -> float:
            toks = tokenize(s)
            if not toks:
                return 0.0
            hits = sum(1 for t in toks if t in top_keywords)
            return hits / len(set(toks))
        ranked = sorted(sentences, key=lambda p: sent_score(p[1]), reverse=True)
        key_points = []
        used = set()
        for spk, s in ranked:
            sig = s[:80]
            if sig in used:
                continue
            used.add(sig)
            if len(s.split()) >= 6:
                key_points.append(f"- {s}")
            if len(key_points) >= 6:
                break

        # Extract dates/times and places heuristically (en/es)
        months = r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december"
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            rf"\b\d{{1,2}}\s+(?:de\s+)?(?:{months})\b",
            rf"\b(?:{months})\s+\d{{1,2}}\b",
            r"\b\d{1,2}:\d{2}\b",
        ]
        place_patterns = [
            r"\b(?:en|in|at)\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ]+){0,2})",
            r"\b(?:oficina|office|sala|meeting|zoom|teams|google\s+meet)\b",
        ]
        dates: Set[str] = set()
        places: Set[str] = set()
        for _, s in sentences:
            for pat in date_patterns:
                for m in re.findall(pat, s, flags=re.IGNORECASE):
                    dates.add(m if isinstance(m, str) else " ".join(m))
            for pat in place_patterns:
                for m in re.findall(pat, s, flags=re.IGNORECASE):
                    if isinstance(m, tuple):
                        m = " ".join([x for x in m if x])
                    places.add(m)

        # Action items / decisions
        decision_markers = [
            'decide','decided','agreed','we will','we\'ll','let\'s','vamos a','acordamos','decidimos','haré','haremos','plan','planear','planteamos'
        ]
        decisions = []
        for spk, s in sentences:
            low = s.lower()
            if any(m in low for m in decision_markers) and len(s.split()) >= 5:
                decisions.append(f"- {s}")
            if len(decisions) >= 5:
                break

        # Build output
        lines: List[str] = []
        lines.append("Conversation Summary\n====================")
        lines.append("")
        # Participants
        lines.append(f"Participants: {len(speaker_stats)}")
        for spk, st in sorted(speaker_stats.items(), key=lambda kv: kv[1]['time'], reverse=True):
            wpm = (st["words"] / (st["time"] / 60)) if st["time"] > 0 else 0
            trait_txt = ", ".join(traits.get(spk, [])) or "neutral"
            lines.append(f"- {spk}: {st['time']:.1f}s speaking, {st['segments']} turns, {wpm:.0f} wpm, traits: {trait_txt}")
        lines.append("")
        # Key points
        if key_points:
            lines.append("Key points:")
            lines.extend(key_points)
            lines.append("")
        # Decisions / Actions
        if decisions:
            lines.append("Decisions / Action items:")
            lines.extend(decisions)
            lines.append("")
        # Dates & Places
        if dates or places:
            lines.append("References:")
            if dates:
                lines.append("- Dates/Times: " + ", ".join(sorted(dates)) )
            if places:
                lines.append("- Places: " + ", ".join(sorted(places)) )
            lines.append("")

        # Trim
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…"
        return text

    def ai_summarize_conversation(self, segments: List[TranscriptSegment],
                                   provider: str = "openai",
                                   model: str = "gpt-4o-mini",
                                   temperature: float = 0.3,
                                   max_tokens: int = 900,
                                   max_chars: int = 16000) -> str:
        """Use an AI model to generate a richer, dynamic summary.
        Requires OPENAI_API_KEY when provider=openai.
        """
        if not segments:
            return "No content to summarize."
        transcript_lines: List[str] = []
        for seg in segments:
            line = f"[{seg.start_time:.1f}-{seg.end_time:.1f}s] {seg.speaker}: {seg.text.strip()}"
            transcript_lines.append(line)
        transcript = "\n".join(transcript_lines)
        if len(transcript) > max_chars:
            logger.warning(f"Transcript length {len(transcript)} exceeds ai_max_chars={max_chars}; truncating.")
            transcript = transcript[:max_chars]

        system_prompt = (
            "You are an expert conversation analyst. Read the transcript and produce a concise, useful summary. "
            "Output must be structured, factual, and non-repetitive."
        )
        user_prompt = (
            "Summarize the following conversation. Provide:\n"
            "1) Participants: count and brief traits (dominant/concise/critical/positive/etc.).\n"
            "2) Summary: 6-10 bullet points of the main ideas and flow.\n"
            "3) Decisions/Action items (if any).\n"
            "4) Dates/Times and Places mentioned (if any).\n"
            "Keep it short but informative. Use the transcript below.\n\n"
            f"Transcript (partial if long):\n{transcript}"
        )

        if provider.lower() == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set; cannot run AI summarization.")
            try:
                if _HAS_OPENAI_SDK:
                    client = OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                    content = resp.choices[0].message.content.strip()
                    return content
                elif _HAS_OPENAI_LEGACY:
                    openai_legacy.api_key = api_key
                    resp = openai_legacy.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                    return resp["choices"][0]["message"]["content"].strip()
                else:
                    raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
            except Exception as e:
                logger.error(f"AI summarization failed: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
    
    def _save_json(self, segments: List[TranscriptSegment], output_path: str):
        """Save as JSON format."""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_segments": len(segments),
                "speakers": list(set(seg.speaker for seg in segments))
            },
            "segments": [asdict(seg) for seg in segments]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    
    def _save_srt(self, segments: List[TranscriptSegment], output_path: str):
        """Save as SRT subtitle format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                f.write(segment.to_srt_format(i) + "\n")
        logger.info(f"SRT file saved to {output_path}")
    
    def _save_txt(self, segments: List[TranscriptSegment], output_path: str):
        """Save as plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Audio Transcription with Speaker Diarization\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for segment in segments:
                timestamp = f"[{segment.start_time:.1f}s-{segment.end_time:.1f}s]"
                f.write(f"{timestamp} {segment.speaker}: {segment.text}\n\n")
        logger.info(f"Text file saved to {output_path}")


def main():
    """Command-line interface for the transcription tool."""
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize audio files using Whisper and pyannote"
    )
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Output file path (without extension)")
    parser.add_argument("-f", "--format", choices=["json", "srt", "txt"], 
                       default="json", help="Output format")
    parser.add_argument("-w", "--whisper-model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("-l", "--language", help="Audio language (auto-detect if not specified)")
    parser.add_argument("--boost-accuracy", action="store_true",
                       help="Use stricter decoding settings for Whisper (temperature=0, no conditioning)")
    parser.add_argument("--segment-padding", type=float, default=0.5,
                       help="Seconds of audio context to pad around each diarized segment")
    parser.add_argument("--enhance", default="off", choices=["off", "light", "medium"],
                       help="Audio enhancement level: off (fast), light (no denoise), medium (adds denoise)")
    parser.add_argument("--similarity-threshold", type=float, default=0.72,
                       help="Cosine similarity threshold for cross-file speaker matching (0-1)")
    parser.add_argument("--min-speakers", type=int, default=1,
                       help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=10,
                       help="Maximum number of speakers")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging verbosity level")
    parser.add_argument("--save-signatures", action="store_true",
                       help="Export speaker signatures (average embeddings) to JSON")
    parser.add_argument("--signatures-file", default="speaker_profiles/signatures.json",
                       help="Path to save signatures JSON (used with --save-signatures)")
    parser.add_argument("--summarize", action="store_true",
                       help="Generate and save a conversation summary")
    parser.add_argument("--summary-file", default="output/summary.txt",
                       help="Path to save conversation summary (used with --summarize)")
    parser.add_argument("--ai-summary", action="store_true",
                       help="Use AI (OpenAI) to generate a higher-quality summary")
    parser.add_argument("--ai-model", default="gpt-4o-mini",
                       help="Model for AI summarization (OpenAI chat)")
    parser.add_argument("--ai-max-tokens", type=int, default=900,
                       help="Max tokens for AI summary")
    parser.add_argument("--ai-temperature", type=float, default=0.3,
                       help="Temperature for AI summary")
    
    args = parser.parse_args()
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s'
    )
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file '{args.audio_file}' not found")
        return
    
    # Generate output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        args.output = f"output/{base_name}_transcription"
    
    # Add appropriate extension
    if args.format == "json" and not args.output.endswith(".json"):
        args.output += ".json"
    elif args.format == "srt" and not args.output.endswith(".srt"):
        args.output += ".srt"
    elif args.format == "txt" and not args.output.endswith(".txt"):
        args.output += ".txt"
    
    try:
        # Initialize transcriber
        transcriber = AudioTranscriberDiarizer(
            whisper_model=args.whisper_model,
            language=args.language,
            boost_accuracy=args.boost_accuracy,
            segment_padding=args.segment_padding,
            enhancement_level=args.enhance,
            similarity_threshold=args.similarity_threshold
        )
        
        # Process audio
        segments = transcriber.process_audio(
            args.audio_file,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        
        if not segments:
            logger.warning("No speech segments detected in the audio file")
            return
        
        # Save results
        transcriber.save_results(segments, args.output, args.format)
        
        # Optionally export signatures JSON
        if args.save_signatures:
            transcriber.save_signatures(args.signatures_file)
        
        # Optionally generate conversation summary
        if args.summarize:
            try:
                if args.ai_summary:
                    summary = transcriber.ai_summarize_conversation(
                        segments,
                        provider="openai",
                        model=args.ai_model,
                        temperature=args.ai_temperature,
                        max_tokens=args.ai_max_tokens,
                    )
                else:
                    summary = transcriber.summarize_conversation(segments)
                os.makedirs(os.path.dirname(args.summary_file) if os.path.dirname(args.summary_file) else ".", exist_ok=True)
                with open(args.summary_file, "w", encoding="utf-8") as f:
                    f.write(summary)
                logger.info(f"Summary saved: {args.summary_file}")
            except Exception as e:
                logger.warning(f"Failed to create summary: {str(e)}")
        
        # Print summary
        total_duration = segments[-1].end_time if segments else 0
        unique_speakers = len(set(seg.speaker for seg in segments))
        
        logger.info("Processing Summary:")
        logger.info(f"- Audio duration: {total_duration:.1f} seconds")
        logger.info(f"- Total segments: {len(segments)}")
        logger.info(f"- Unique speakers: {unique_speakers}")
        logger.info(f"- Output saved: {args.output}")

        # Final metrics
        m = getattr(transcriber, 'metrics', {})
        if m:
            logger.info("Timing Metrics:")
            logger.info(f"- Load audio: {m.get('load_audio_s', 0):.2f}s")
            logger.info(f"- Enhancement: {m.get('enhance_s', 0):.2f}s (level={getattr(transcriber, 'enhancement_level', 'off')})")
            logger.info(f"- Language detect: {m.get('language_detect_s', 0):.2f}s")
            logger.info(f"- Diarization: {m.get('diarization_s', 0):.2f}s (speakers={int(m.get('speakers_detected', 0))})")
            logger.info(f"- Profiles mapping: {m.get('profiles_map_s', 0):.2f}s (threshold={getattr(transcriber, 'similarity_threshold', 0.72):.2f})")
            logger.info(f"- Transcription: {m.get('transcription_s', 0):.2f}s (segments={int(m.get('segments', 0))})")
            logger.info(f"- Total process_audio: {m.get('process_audio_s', 0):.2f}s")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")


# Example usage functions
def example_basic_usage():
    """Example of basic usage programmatically."""
    audio_file = "sample_audio.wav"  # Replace with your audio file
    
    # Initialize the transcriber
    transcriber = AudioTranscriberDiarizer(
        whisper_model="base",
        language="en"  # Set to None for auto-detection
    )
    
    # Process the audio
    segments = transcriber.process_audio(audio_file)
    
    # Save in multiple formats
    transcriber.save_results(segments, "output/transcript.json", "json")
    transcriber.save_results(segments, "output/transcript.srt", "srt")
    transcriber.save_results(segments, "output/transcript.txt", "txt")
    
    return segments


def example_advanced_usage():
    """Example with more advanced configuration."""
    audio_file = "meeting_recording.wav"
    
    # Use a larger Whisper model for better accuracy
    transcriber = AudioTranscriberDiarizer(
        whisper_model="large",
        language=None  # Auto-detect language
    )
    
    # Process with specific speaker constraints
    segments = transcriber.process_audio(
        audio_file,
        min_speakers=2,  # Expect at least 2 speakers
        max_speakers=5   # But no more than 5
    )
    
    # Filter out short segments (less than 1 second)
    filtered_segments = [seg for seg in segments if seg.duration() >= 1.0]
    
    # Save filtered results
    transcriber.save_results(filtered_segments, "output/meeting_transcript.json")
    
    return filtered_segments


def analyze_conversation(segments: List[TranscriptSegment]):
    """Analyze the conversation for insights."""
    if not segments:
        return
    
    # Speaker statistics
    speaker_stats = {}
    for segment in segments:
        if segment.speaker not in speaker_stats:
            speaker_stats[segment.speaker] = {
                "total_time": 0,
                "segment_count": 0,
                "word_count": 0
            }
        
        stats = speaker_stats[segment.speaker]
        stats["total_time"] += segment.duration()
        stats["segment_count"] += 1
        stats["word_count"] += len(segment.text.split())
    
    print("\nConversation Analysis:")
    print("-" * 30)
    
    for speaker, stats in speaker_stats.items():
        avg_segment_length = stats["total_time"] / stats["segment_count"]
        words_per_minute = stats["word_count"] / (stats["total_time"] / 60)
        
        print(f"{speaker}:")
        print(f"  Speaking time: {stats['total_time']:.1f}s")
        print(f"  Segments: {stats['segment_count']}")
        print(f"  Words: {stats['word_count']}")
        print(f"  Avg segment: {avg_segment_length:.1f}s")
        print(f"  Speaking rate: {words_per_minute:.1f} words/min")
        print()


if __name__ == "__main__":
    # Check if running as script with arguments
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Run example if no arguments provided
        logger.info("Running example usage...")
        logger.info("Use --help to see command-line options")
        
        # You can uncomment these to run examples:
        # segments = example_basic_usage()
        # analyze_conversation(segments)