"""Tools for AI Book Composer."""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
import ffmpeg
from faster_whisper import WhisperModel

from ..config import settings


class FileListingTool:
    """Tool to list files in a directory."""
    
    name = "list_files"
    description = "List all files in the input directory"
    
    def __init__(self, input_directory: str):
        self.input_directory = Path(input_directory)
    
    def run(self, **kwargs) -> List[Dict[str, str]]:
        """List files in the directory.
        
        Returns:
            List of file information dictionaries
        """
        files = []
        for file_path in self.input_directory.rglob("*"):
            if file_path.is_file():
                files.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size
                })
        return files


class TextFileReaderTool:
    """Tool to read text file content with line range support."""
    
    name = "read_text_file"
    description = "Read content from a text file (up to 100 lines at a time)"
    
    def run(self, file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Read text file content.
        
        Args:
            file_path: Path to the text file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive), None for up to max_lines
            
        Returns:
            Dictionary with content and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            start_idx = max(0, start_line - 1)
            
            if end_line is None:
                end_idx = min(start_idx + settings.max_lines_per_read, total_lines)
            else:
                end_idx = min(end_line, total_lines)
            
            content = ''.join(lines[start_idx:end_idx])
            
            return {
                "content": content,
                "start_line": start_line,
                "end_line": end_idx,
                "total_lines": total_lines,
                "has_more": end_idx < total_lines
            }
        except Exception as e:
            return {
                "error": str(e),
                "content": ""
            }


class AudioTranscriptionTool:
    """Tool to transcribe audio files using ffmpeg and faster-whisper."""
    
    name = "transcribe_audio"
    description = "Transcribe an audio file to text"
    
    def __init__(self, model_size: str = "base"):
        """Initialize transcription tool.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def run(self, file_path: str) -> Dict[str, Any]:
        """Transcribe audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription and metadata
        """
        try:
            segments, info = self.model.transcribe(file_path, beam_size=5)
            
            transcription = []
            for segment in segments:
                transcription.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
            
            full_text = " ".join([seg["text"] for seg in transcription])
            
            return {
                "transcription": full_text,
                "segments": transcription,
                "language": info.language,
                "duration": info.duration
            }
        except Exception as e:
            return {
                "error": str(e),
                "transcription": ""
            }


class VideoTranscriptionTool:
    """Tool to transcribe video files by extracting audio and using Whisper."""
    
    name = "transcribe_video"
    description = "Transcribe a video file to text"
    
    def __init__(self, model_size: str = "base"):
        """Initialize transcription tool.
        
        Args:
            model_size: Whisper model size
        """
        self.audio_tool = AudioTranscriptionTool(model_size)
    
    def run(self, file_path: str) -> Dict[str, Any]:
        """Transcribe video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dictionary with transcription and metadata
        """
        try:
            # Extract audio from video to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_path = tmp_file.name
            
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Transcribe audio
            result = self.audio_tool.run(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "transcription": ""
            }


class ChapterWriterTool:
    """Tool to write chapters to files."""
    
    name = "write_chapter"
    description = "Write a chapter to a file"
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def run(self, chapter_number: int, title: str, content: str) -> Dict[str, Any]:
        """Write chapter to file.
        
        Args:
            chapter_number: Chapter number
            title: Chapter title
            content: Chapter content
            
        Returns:
            Dictionary with file path and status
        """
        try:
            # Sanitize title for filename - remove invalid characters
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
            safe_title = safe_title.replace(' ', '_')
            filename = f"chapter_{chapter_number:02d}_{safe_title}.txt"
            file_path = self.output_directory / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Chapter {chapter_number}: {title}\n\n")
                f.write(content)
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chapter_number": chapter_number,
                "title": title
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ChapterListWriterTool:
    """Tool to write list of chapters."""
    
    name = "write_chapter_list"
    description = "Write the list of planned chapters"
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def run(self, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write chapter list to file.
        
        Args:
            chapters: List of chapter dictionaries with 'number', 'title', 'description'
            
        Returns:
            Dictionary with file path and status
        """
        try:
            file_path = self.output_directory / "chapter_list.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chapters, f, indent=2)
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chapter_count": len(chapters)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
