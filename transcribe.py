#!/usr/bin/env python3
import os
import json
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from moviepy.editor import VideoFileClip
from pathlib import Path
import warnings
from datetime import timedelta
import logging
import argparse
import sys
import librosa

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class VideoTranscriber:
    def __init__(self, device=None, models_dir="./models"):
        """
        Initialize the transcriber with Whisper model.
        
        Args:
            device: Device to use (cuda, cpu, or None for auto-detection)
            models_dir: Directory containing downloaded models
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Set up models directory
        self.models_dir = Path(models_dir)
        
        # Load Whisper model from local files
        logger.info(f"Loading Whisper model from local files...")
        self.whisper_model, self.whisper_processor = self._load_whisper_from_files()
    
    def _load_whisper_from_files(self):
        """Load Whisper model from manually downloaded files."""
        whisper_dir = self.models_dir / "whisper-large-v3"
        
        if not whisper_dir.exists():
            raise FileNotFoundError(
                f"Whisper model files not found in {whisper_dir}\n"
                "Please download the model files from:\n"
                "https://huggingface.co/openai/whisper-large-v3"
            )
        
        try:
            # Load processor and model using transformers
            processor = WhisperProcessor.from_pretrained(str(whisper_dir), local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(
                str(whisper_dir), 
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                model = model.cuda()
            
            logger.info("✓ Whisper model loaded successfully from local files")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.error("Make sure you have downloaded all required files:")
            logger.error("- model.safetensors")
            logger.error("- config.json")
            logger.error("- tokenizer.json")
            logger.error("- preprocessor_config.json")
            logger.error("- vocabulary.json")
            logger.error("- merges.txt")
            raise
    
    def extract_audio(self, video_path, audio_path):
        """Extract audio from video file."""
        logger.info(f"Extracting audio from {video_path}")
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path), codec='pcm_s16le', logger=None)
        video.close()
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        logger.info("Transcribing audio with Whisper large-v3...")
        
        # Load and preprocess audio
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        # Process in chunks for long audio
        chunk_length = 30  # seconds
        chunk_samples = chunk_length * sr
        chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
        
        all_text = []
        all_segments = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Process audio chunk
            inputs = self.whisper_processor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.whisper_model.generate(
                    inputs["input_features"],
                    max_length=448,
                    num_beams=5,
                    temperature=0.0
                )
            
            # Decode transcription
            transcription = self.whisper_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Calculate timestamps
            start_time = i * chunk_length
            end_time = min((i + 1) * chunk_length, len(audio) / sr)
            
            if transcription.strip():  # Only add non-empty segments
                all_text.append(transcription)
                all_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': transcription.strip()
                })
        
        # Return in format compatible with output
        return {
            'text': ' '.join(all_text),
            'segments': all_segments,
            'language': 'en',  # You might want to detect this
            'duration': len(audio) / sr
        }
    
    def format_time(self, seconds):
        """Convert seconds to readable time format."""
        return str(timedelta(seconds=seconds)).split('.')[0]
    
    def process_video(self, video_path, output_dir):
        """Process a single video file."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Create temporary audio file
        audio_path = output_dir / f"{video_path.stem}_audio.wav"
        
        try:
            # Extract audio
            self.extract_audio(video_path, audio_path)
            
            # Transcribe
            transcription = self.transcribe_audio(audio_path)
            
            # Prepare output
            output = {
                'video_file': video_path.name,
                'duration': transcription.get('duration', 0),
                'language': transcription.get('language', 'unknown'),
                'segments': []
            }
            
            # Format output
            for segment in transcription['segments']:
                output['segments'].append({
                    'start_time': self.format_time(segment['start']),
                    'end_time': self.format_time(segment['end']),
                    'text': segment['text']
                })
            
            # Save JSON output
            json_output_path = output_dir / f"{video_path.stem}_transcription.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            # Save human-readable output
            txt_output_path = output_dir / f"{video_path.stem}_transcription.txt"
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription for: {video_path.name}\n")
                f.write(f"Duration: {self.format_time(output['duration'])}\n")
                f.write(f"Language: {output['language']}\n")
                f.write("=" * 80 + "\n\n")
                
                for segment in output['segments']:
                    f.write(f"[{segment['start_time']} - {segment['end_time']}]\n")
                    f.write(f"{segment['text']}\n\n")
            
            logger.info(f"Transcription saved to:")
            logger.info(f"  - {json_output_path}")
            logger.info(f"  - {txt_output_path}")
            
        finally:
            # Clean up temporary audio file
            if audio_path.exists():
                audio_path.unlink()
    
    def process_all_videos(self, input_dir, output_dir):
        """Process all video files in the input directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.MOV', '.MP4']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Process each video
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\nProcessing video {i}/{len(video_files)}: {video_file.name}")
            try:
                self.process_video(video_file, output_dir)
                logger.info(f"✓ Successfully processed {video_file.name}")
            except Exception as e:
                logger.error(f"✗ Error processing {video_file.name}: {str(e)}")
                continue
        
        logger.info("\n✓ All videos processed!")

def main():
    parser = argparse.ArgumentParser(description='Transcribe videos using Whisper large-v3')
    parser.add_argument('--input', '-i', default='./input', help='Input directory containing video files')
    parser.add_argument('--output', '-o', default='./output', help='Output directory for transcriptions')
    parser.add_argument('--models-dir', default='./models', help='Directory containing downloaded models')
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Check if models directory exists
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        logger.error("Please download the Whisper model files first.")
        logger.error("\nDownload instructions:")
        logger.error("1. Create models directory: mkdir -p models/whisper-large-v3")
        logger.error("2. Download files from: https://huggingface.co/openai/whisper-large-v3")
        logger.error("   - model.safetensors")
        logger.error("   - config.json")
        logger.error("   - tokenizer.json")
        logger.error("   - preprocessor_config.json")
        logger.error("   - vocabulary.json")
        logger.error("   - merges.txt")
        sys.exit(1)
    
    # Initialize transcriber
    logger.info("Initializing Whisper transcriber...")
    try:
        transcriber = VideoTranscriber(
            device=args.device,
            models_dir=args.models_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize transcriber: {e}")
        sys.exit(1)
    
    # Process all videos
    transcriber.process_all_videos(args.input, args.output)

if __name__ == "__main__":
    main()