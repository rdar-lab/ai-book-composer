"""Enhanced tools with security, logging, and additional format support."""

import os
import json
import tempfile
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
import ffmpeg
from faster_whisper import WhisperModel

# Document processing
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from striprtf.striprtf import rtf_to_text

from ..config import settings
from ..logging_config import logger


def is_path_safe(base_path: Path, target_path: Path) -> bool:
    """Check if target path is within base path (prevent directory traversal).
    
    Args:
        base_path: Base directory
        target_path: Target path to check
        
    Returns:
        True if safe, False otherwise
    """
    if settings.security.allow_directory_traversal:
        return True
    
    try:
        base_abs = base_path.resolve()
        target_abs = target_path.resolve()
        return target_abs.is_relative_to(base_abs)
    except (ValueError, OSError):
        return False


def check_file_size(file_path: Path) -> bool:
    """Check if file size is within limits.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if within limits, False otherwise
    """
    max_size_bytes = settings.security.max_file_size_mb * 1024 * 1024
    file_size = file_path.stat().st_size
    
    if file_size > max_size_bytes:
        logger.warning(f"File {file_path} exceeds size limit: {file_size} > {max_size_bytes}")
        return False
    return True


class FileListingTool:
    """Tool to list files in a directory with security checks."""
    
    name = "list_files"
    description = "List all files in the input directory"
    
    def __init__(self, input_directory: str):
        self.input_directory = Path(input_directory).resolve()
        logger.info(f"FileListingTool initialized for directory: {self.input_directory}")
    
    def run(self, **kwargs) -> List[Dict[str, str]]:
        """List files in the directory.
        
        Returns:
            List of file information dictionaries
        """
        logger.debug(f"Listing files in {self.input_directory}")
        files = []
        
        try:
            for file_path in self.input_directory.rglob("*"):
                if file_path.is_file():
                    # Security check
                    if not is_path_safe(self.input_directory, file_path):
                        logger.warning(f"Skipping file outside base directory: {file_path}")
                        continue
                    
                    files.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "extension": file_path.suffix,
                        "size": file_path.stat().st_size
                    })
            
            logger.info(f"Found {len(files)} files")
            return files
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise


class TextFileReaderTool:
    """Tool to read text files with support for multiple formats."""
    
    name = "read_text_file"
    description = "Read content from text files (txt, md, rst, docx, rtf, pdf)"
    
    def run(self, file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Read text file content.
        
        Args:
            file_path: Path to the text file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive), None for up to max_lines
            
        Returns:
            Dictionary with content and metadata
        """
        file_path = Path(file_path).resolve()
        logger.debug(f"Reading file: {file_path}, lines {start_line} to {end_line}")
        
        # Security checks
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found", "content": ""}
        
        if not check_file_size(file_path):
            return {"error": "File too large", "content": ""}
        
        try:
            # Extract text based on file type
            extension = file_path.suffix.lower()
            
            if extension in ['.txt', '.md', '.rst']:
                content = self._read_plain_text(file_path)
            elif extension == '.docx':
                content = self._read_docx(file_path)
            elif extension == '.rtf':
                content = self._read_rtf(file_path)
            elif extension == '.pdf':
                content = self._read_pdf(file_path)
            else:
                logger.warning(f"Unsupported file format: {extension}")
                return {"error": f"Unsupported format: {extension}", "content": ""}
            
            # Split into lines and apply range
            lines = content.split('\n')
            total_lines = len(lines)
            start_idx = max(0, start_line - 1)
            
            if end_line is None:
                end_idx = min(start_idx + settings.text_reading.max_lines_per_read, total_lines)
            else:
                end_idx = min(end_line, total_lines)
            
            selected_content = '\n'.join(lines[start_idx:end_idx])
            
            logger.info(f"Read {end_idx - start_idx} lines from {file_path}")
            
            return {
                "content": selected_content,
                "start_line": start_line,
                "end_line": end_idx,
                "total_lines": total_lines,
                "has_more": end_idx < total_lines
            }
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"error": str(e), "content": ""}
    
    def _read_plain_text(self, file_path: Path) -> str:
        """Read plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file."""
        doc = DocxDocument(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    def _read_rtf(self, file_path: Path) -> str:
        """Read RTF file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content)
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file."""
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return '\n'.join(text_parts)


class AudioTranscriptionTool:
    """Tool to transcribe audio files with support for local and remote Whisper."""
    
    name = "transcribe_audio"
    description = "Transcribe audio files (mp3, wav, m4a, flac, ogg)"
    
    def __init__(self):
        """Initialize transcription tool."""
        self.mode = settings.whisper.mode
        logger.info(f"AudioTranscriptionTool initialized in {self.mode} mode")
        
        if self.mode == "local":
            model_size = settings.whisper.model_size
            device = settings.whisper.local["device"]
            compute_type = settings.whisper.local["compute_type"]
            logger.info(f"Loading Whisper model: {model_size} on {device}")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        else:
            self.endpoint = settings.whisper.remote["endpoint"]
            self.api_key = settings.whisper.remote["api_key"]
            logger.info(f"Using remote Whisper at {self.endpoint}")
    
    def run(self, file_path: str) -> Dict[str, Any]:
        """Transcribe audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription and metadata
        """
        file_path = Path(file_path).resolve()
        logger.info(f"Transcribing audio file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return {"error": "File not found", "transcription": ""}
        
        if not check_file_size(file_path):
            return {"error": "File too large", "transcription": ""}
        
        try:
            if self.mode == "local":
                return self._transcribe_local(file_path)
            else:
                return self._transcribe_remote(file_path)
        except Exception as e:
            logger.error(f"Error transcribing audio {file_path}: {e}")
            return {"error": str(e), "transcription": ""}
    
    def _transcribe_local(self, file_path: Path) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        segments, info = self.model.transcribe(str(file_path), beam_size=5)
        
        transcription = []
        for segment in segments:
            transcription.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        full_text = " ".join([seg["text"] for seg in transcription])
        logger.info(f"Transcription complete: {len(transcription)} segments, {info.duration}s")
        
        return {
            "transcription": full_text,
            "segments": transcription,
            "language": info.language,
            "duration": info.duration
        }
    
    def _transcribe_remote(self, file_path: Path) -> Dict[str, Any]:
        """Transcribe using remote Whisper service."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.endpoint}/transcribe",
                files=files,
                headers=headers,
                timeout=600
            )
            response.raise_for_status()
            result = response.json()
        
        logger.info(f"Remote transcription complete")
        return result


class VideoTranscriptionTool:
    """Tool to transcribe video files with chunking support for large files."""
    
    name = "transcribe_video"
    description = "Transcribe video files (mp4, avi, mov, mkv)"
    
    def __init__(self):
        """Initialize transcription tool."""
        self.audio_tool = AudioTranscriptionTool()
        self.chunk_duration = settings.media_processing.chunk_duration
        logger.info(f"VideoTranscriptionTool initialized with {self.chunk_duration}s chunks")
    
    def run(self, file_path: str) -> Dict[str, Any]:
        """Transcribe video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dictionary with transcription and metadata
        """
        file_path = Path(file_path).resolve()
        logger.info(f"Transcribing video file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"Video file not found: {file_path}")
            return {"error": "File not found", "transcription": ""}
        
        if not check_file_size(file_path):
            return {"error": "File too large", "transcription": ""}
        
        try:
            # Get video duration
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            logger.info(f"Video duration: {duration}s")
            
            # Check if we need to chunk
            if duration > self.chunk_duration:
                return self._transcribe_chunked(file_path, duration)
            else:
                return self._transcribe_single(file_path)
        except Exception as e:
            logger.error(f"Error transcribing video {file_path}: {e}")
            return {"error": str(e), "transcription": ""}
    
    def _transcribe_single(self, file_path: Path) -> Dict[str, Any]:
        """Transcribe entire video at once."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_path = tmp_file.name
        
        try:
            # Extract audio
            stream = ffmpeg.input(str(file_path))
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Transcribe
            result = self.audio_tool.run(audio_path)
            return result
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def _transcribe_chunked(self, file_path: Path, duration: float) -> Dict[str, Any]:
        """Transcribe video in chunks."""
        logger.info(f"Transcribing video in chunks of {self.chunk_duration}s")
        
        all_segments = []
        num_chunks = int(duration / self.chunk_duration) + 1
        
        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            chunk_duration = min(self.chunk_duration, duration - start_time)
            
            logger.debug(f"Processing chunk {i+1}/{num_chunks}: {start_time}s-{start_time+chunk_duration}s")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_path = tmp_file.name
            
            try:
                # Extract audio chunk
                stream = ffmpeg.input(str(file_path), ss=start_time, t=chunk_duration)
                stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                # Transcribe chunk
                result = self.audio_tool.run(audio_path)
                
                if "segments" in result:
                    # Adjust timestamps
                    for seg in result["segments"]:
                        seg["start"] += start_time
                        seg["end"] += start_time
                    all_segments.extend(result["segments"])
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        
        full_text = " ".join([seg["text"] for seg in all_segments])
        logger.info(f"Video transcription complete: {len(all_segments)} segments")
        
        return {
            "transcription": full_text,
            "segments": all_segments,
            "language": all_segments[0].get("language") if all_segments else "unknown",
            "duration": duration
        }


class ChapterWriterTool:
    """Tool to write chapters to files."""
    
    name = "write_chapter"
    description = "Write a chapter to a file"
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChapterWriterTool initialized for directory: {self.output_directory}")
    
    def run(self, chapter_number: int, title: str, content: str) -> Dict[str, Any]:
        """Write chapter to file.
        
        Args:
            chapter_number: Chapter number
            title: Chapter title
            content: Chapter content
            
        Returns:
            Dictionary with file path and status
        """
        logger.debug(f"Writing chapter {chapter_number}: {title}")
        
        try:
            # Sanitize title for filename
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
            safe_title = safe_title.replace(' ', '_')
            filename = f"chapter_{chapter_number:02d}_{safe_title}.txt"
            file_path = self.output_directory / filename
            
            # Security check
            if not is_path_safe(self.output_directory, file_path):
                logger.error(f"Invalid output path: {file_path}")
                return {"success": False, "error": "Invalid output path"}
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Chapter {chapter_number}: {title}\n\n")
                f.write(content)
            
            logger.info(f"Chapter {chapter_number} written to {file_path}")
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chapter_number": chapter_number,
                "title": title
            }
        except Exception as e:
            logger.error(f"Error writing chapter: {e}")
            return {"success": False, "error": str(e)}


class ChapterListWriterTool:
    """Tool to write list of chapters."""
    
    name = "write_chapter_list"
    description = "Write the list of planned chapters"
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChapterListWriterTool initialized for directory: {self.output_directory}")
    
    def run(self, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write chapter list to file.
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            Dictionary with file path and status
        """
        logger.debug(f"Writing chapter list with {len(chapters)} chapters")
        
        try:
            file_path = self.output_directory / "chapter_list.json"
            
            # Security check
            if not is_path_safe(self.output_directory, file_path):
                logger.error(f"Invalid output path: {file_path}")
                return {"success": False, "error": "Invalid output path"}
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chapters, f, indent=2)
            
            logger.info(f"Chapter list written to {file_path}")
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chapter_count": len(chapters)
            }
        except Exception as e:
            logger.error(f"Error writing chapter list: {e}")
            return {"success": False, "error": str(e)}
