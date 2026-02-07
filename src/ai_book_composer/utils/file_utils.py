"""Enhanced tools with security, logging, and additional format support."""

import base64
import hashlib
import json
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
from typing import Optional

import ffmpeg
import requests
from PIL import Image
from docx import Document as DocxDocument
from faster_whisper import WhisperModel
from langchain_core.messages import HumanMessage
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text

from ..config import Settings
from ..llm import get_llm

logger = logging.getLogger(__name__)


def is_path_safe(settings: Settings, base_path: Path, target_path: Path) -> bool:
    """Check if target path is within base path (prevent directory traversal).

    Args:
        settings: The project settings
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


def is_file_size_within_limits(settings: Settings, file_path: Path) -> bool:
    """Check if file size is within limits.

    Args:
        settings: The project settings
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


def list_input_files(settings: Settings, input_directory_str: str):
    """List files in the directory.

    Returns:
        List of file information dictionaries
    """

    input_directory = Path(input_directory_str).resolve()

    logger.debug(f"Listing files in {input_directory}")
    files = []

    try:
        for file_path in input_directory.rglob("*"):
            if file_path.is_file():
                # Security check
                if not is_path_safe(settings, input_directory, file_path):
                    raise Exception(f"file outside base directory: {file_path}")

                files.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size
                })

        logger.info(f"Found {len(files)} files")
        return files
    except Exception as e:
        logger.exception(f"Error listing files: {e}")
        raise


def read_text_file(settings: Settings, file_path: str) -> Dict[str, Any]:
    """Read text file content.

    Args:
        settings: The project settings
        file_path: Path to the text file
    Returns:
        Dictionary with content and metadata
    """
    file_path = Path(file_path).resolve()
    logger.debug(f"Reading file: {file_path}")

    # Security checks
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise Exception("File not found")

    if not is_file_size_within_limits(settings, file_path):
        raise Exception("File too large")

    try:
        # Extract text based on file type
        extension = file_path.suffix.lower()

        if extension in ['.txt', '.md', '.rst']:
            content = _read_plain_text(file_path)
        elif extension == '.docx':
            content = _read_docx(file_path)
        elif extension == '.rtf':
            content = _read_rtf(file_path)
        elif extension == '.pdf':
            content = _read_pdf(file_path)
        else:
            raise Exception(f"Unsupported format: {extension}")

        return {
            "content": content
        }
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        raise


def _read_plain_text(file_path: Path) -> str:
    """Read plain text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _read_docx(file_path: Path) -> str:
    """Read DOCX file."""
    doc = DocxDocument(str(file_path))
    return '\n'.join([para.text for para in doc.paragraphs])


def _read_rtf(file_path: Path) -> str:
    """Read RTF file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        rtf_content = f.read()
    return rtf_to_text(rtf_content)


def _read_pdf(file_path: Path) -> str:
    """Read PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())
    return '\n'.join(text_parts)


def read_audio_file(settings: Settings, file_path: str, language: Optional[str] = None, cache_results=True) -> Dict[
    str, Any]:
    """Transcribe audio file.

    Args:
        settings: The project settings
        file_path: Path to the audio file
        language: Optional language code (e.g., 'en', 'he' for Hebrew). If None, auto-detects.
        cache_results: Whether to cache transcription results

    Returns:
        Dictionary with transcription and metadata
    """

    mode = settings.whisper.mode
    logger.info(f"AudioTranscriptionTool initialized in {mode} mode")

    file_path = Path(file_path).resolve()
    logger.info(f"Transcribing audio file: {file_path}, language: {language or 'auto-detect'}")

    if not file_path.exists():
        logger.error(f"Audio file not found: {file_path}")
        raise Exception("File not found")

    if not is_file_size_within_limits(settings, file_path):
        raise Exception("File too large")

    # Check for cached transcription (including language in cache key)
    cache_path = get_cache_path(settings, file_path, language=language)
    cached_result = read_cache(cache_path)
    if cache_results and cached_result is not None:
        return cached_result

    try:
        # Replace any 'en..' with simple 'en'
        if language and 'en' in language.lower():
            language = 'en'

        if mode == "local":
            result = transcribe_local(settings, file_path, language)
        else:
            result = transcribe_remote(settings, file_path, language)

        if cache_results:
            write_cache(cache_path, result)

        return result
    except Exception as e:
        logger.exception(f"Error transcribing audio {file_path}: {e}")
        raise


def get_cache_path(settings: Settings, file_path: Path | str, prefix: str = '', ext: str = 'json',
                   language: Optional[str] = None) -> Path:
    """Get cache file path for a given file.

    Args:
        settings: The project settings
        file_path: Path to the file
        prefix: prefix to add to the
        ext: the file extension to add
        language: Language code used for transcription (affects cache key)

    Returns:
        Path to the cache file
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    cache_dir = Path(settings.general.cache_dir)

    # Include language in cache filename to handle different language transcriptions
    lang_suffix = f"_{language}" if language else ""
    cache_filename = f".{prefix}{_generate_unique_name_for_path(file_path)}{lang_suffix}.{ext}"

    # Create a .cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / cache_filename


def read_cache(cache_path: Path) -> Optional[Any]:
    """Read cached transcription if it exists.

    Args:
        cache_path: Path to the cache file

    Returns:
        Cached transcription data or None
    """
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Using cache from {cache_path}")
            logger.info(f"Cache hit: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_path}: {e}")

    logger.info(f"Cache miss: {cache_path}")
    return None


def write_cache(cache_path: Path, data: Any) -> None:
    """Write transcription result to cache.

    Args:
        cache_path: Path to the cache file
        data: Transcription data to cache
    """
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached written to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to write cache {cache_path}: {e}")


def transcribe_local(settings: Settings, file_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe using local Whisper model.

    Args:
        settings: The project settings
        file_path: Path to the audio file
        language: Optional language code for transcription
    """
    # Prepare transcription parameters
    model_size = settings.whisper.model_size
    device = settings.whisper.local["device"]
    compute_type = settings.whisper.local["compute_type"]
    logger.info(f"Loading Whisper model: {model_size} on {device}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    transcribe_kwargs: dict[str, Any] = {"beam_size": 5}

    # This is disabled because we can have many source languages
    # if language:
    #    transcribe_kwargs["language"] = language

    segments, info = model.transcribe(str(file_path), **transcribe_kwargs)

    transcription = []
    for segment in segments:
        transcription.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })

    full_text = " ".join([seg["text"] for seg in transcription])
    logger.info(
        f"Transcription complete: {len(transcription)} segments, {info.duration}s, language: {info.language}")

    return {
        "transcription": full_text,
        "segments": transcription,
        "language": info.language,
        "duration": info.duration
    }


def transcribe_remote(settings: Settings, file_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe using remote Whisper service.

    Args:
        settings: The project settings
        file_path: Path to the audio file
        language: Optional language code for transcription
    """
    endpoint = settings.whisper.remote["endpoint"]
    api_key = settings.whisper.remote["api_key"]
    logger.info(f"Using remote Whisper at {endpoint}")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {}

        # This is disabled because we can have many source languages
        # if language:
        #    data['language'] = language

        response = requests.post(
            f"{endpoint}/transcribe",
            files=files,
            data=data,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()

    logger.info(f"Remote transcription complete")
    return result


def read_video_file(settings: Settings, file_path: str, language: Optional[str] = None, cache_results=True) -> Dict[
    str, Any]:
    """Transcribe video file.

    Args:
        settings: The project settings
        file_path: Path to the video file
        language: Optional language code (e.g., 'en', 'he' for Hebrew). If None, auto-detects.
        cache_results: Whether to cache transcription results
    """

    file_path = Path(file_path).resolve()
    logger.info(f"Transcribing video file: {file_path}, language: {language or 'auto-detect'}")

    if not file_path.exists():
        logger.error(f"Video file not found: {file_path}")
        raise Exception("File not found")

    if not is_file_size_within_limits(settings, file_path):
        raise Exception("File too large")

    # Check for cached transcription (including language in cache key)
    cache_path = get_cache_path(settings, file_path, language=language)
    cached_result = read_cache(cache_path)
    if cache_results and cached_result is not None:
        return cached_result

    try:
        # Get video duration
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe['format']['duration'])
        logger.info(f"Video duration: {duration}s")

        result = _transcribe_single(settings, file_path, language)

        if cache_results:
            write_cache(cache_path, result)

        return result
    except Exception as e:
        logger.exception(f"Error transcribing video {file_path}: {e}")
        raise


def _transcribe_single(settings: Settings, file_path: Path, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe entire video at once.

    Args:
        settings: The project settings
        file_path: Path to the video file
        language: Optional language code for transcription
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        audio_path = tmp_file.name

    try:
        # Extract audio
        stream = ffmpeg.input(str(file_path))
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        result = read_audio_file(settings, audio_path, language, cache_results=False)
        return result
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def _generate_unique_name_for_path(base_path: Path):
    # Generate a unique simple name for the base path. The name is unique even if the same filename exists in multiple directories

    file_name = str(base_path.resolve())

    file_simple_name = str(base_path.name).replace(',', '_').replace(' ', '_')

    # Generate hash (MD5) based on full path
    md5_hash = hashlib.md5()
    md5_hash.update(file_name.encode('utf-8'))
    file_hash = md5_hash.hexdigest()

    # Base64 encode the file name to create a unique identifier
    encoded_name = f'{file_simple_name}_{file_hash}'

    return encoded_name


def is_image_meaningful(image_bytes: bytes, black_threshold: float = 0.95) -> bool:
    """Check if an image is meaningful (not just black or empty).
    
    Args:
        image_bytes: Raw image bytes
        black_threshold: Percentage threshold (0.0-1.0) of black/dark pixels to consider image as black
        
    Returns:
        True if image is meaningful, False if it's mostly black/empty
    """
    try:

        # Load image
        img = Image.open(BytesIO(image_bytes))

        # Convert to RGB if needed
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # Convert to grayscale for easier analysis
        if img.mode == 'RGB':
            img = img.convert('L')

        # Resize to smaller size for faster processing (e.g., 100x100).
        # Note: thumbnail() downsamples the image in-place, so the analysis
        # below is performed on this lower-resolution version rather than
        # the original full-size image.
        img.thumbnail((100, 100))

        # Get pixel data - using load() to avoid deprecation warning
        width, height = img.size
        pixels_accessor = img.load()
        total_pixels = width * height

        if total_pixels == 0:
            return False

        # Count dark pixels (below threshold of 30 out of 255).
        # The value 30 was chosen empirically: pixels below this are
        # considered black/very dark for detecting completely black images.
        dark_pixels = 0
        for y in range(height):
            for x in range(width):
                if pixels_accessor[x, y] < 30:
                    dark_pixels += 1

        dark_ratio = dark_pixels / total_pixels

        # Image is meaningful if less than black_threshold of pixels are dark
        return dark_ratio < black_threshold

    except Exception as e:
        logger.warning(f"Error checking if image is meaningful: {e}")
        # If we can't determine, assume it's meaningful to be safe
        return True


def extract_images_from_pdf(settings: Settings, file_path: str) -> Dict[str, Any]:
    """Extract images from a PDF file.

    Args:
        settings: The project settings
        file_path: Path to the PDF file
    """

    try:
        # noinspection PyPackageRequirements
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Cannot extract images from PDF.")

        # Do not fail it, just ignore
        return {"error": "PyMuPDF not installed", "images": []}

    file_path = Path(file_path).resolve()
    logger.info(f"Extracting images from PDF: {file_path}")

    if not file_path.exists():
        logger.error(f"PDF file not found: {file_path}")
        raise Exception("File not found")

    if not is_file_size_within_limits(settings, file_path):
        raise Exception("File too large")

    # Prepare output directory
    output_directory_path = Path(settings.general.cache_dir) / _generate_unique_name_for_path(file_path)
    if not is_path_safe(settings, Path(settings.general.cache_dir), output_directory_path):
        raise Exception(f"unsafe image path: {output_directory_path}")
    output_directory_path.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(file_path)
        extracted_images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Check if image is meaningful (not black)
                if not is_image_meaningful(image_bytes):
                    logger.debug(f"Skipping black/empty image on page {page_num + 1}, index {img_index + 1}")
                    continue

                # Create unique filename
                image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                image_path = output_directory_path / image_filename

                # Security check
                if not is_path_safe(settings, output_directory_path, image_path):
                    raise Exception(f"unsafe image path: {image_path}")

                # Check image size
                image_size_mb = len(image_bytes) / (1024 * 1024)
                if image_size_mb > settings.image_processing.max_image_size_mb:
                    logger.warning(
                        f"Image too large: {image_size_mb}MB > {settings.image_processing.max_image_size_mb}MB")
                    continue

                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                extracted_images.append({
                    "path": str(image_path),
                    "filename": image_filename,
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "format": image_ext,
                    "source_file": str(file_path)
                })

                logger.debug(f"Extracted image: {image_filename}")

        doc.close()
        logger.info(f"Extracted {len(extracted_images)} images from {file_path}")

        return {
            "success": True,
            "images": extracted_images,
            "count": len(extracted_images)
        }
    except Exception as e:
        logger.exception(f"Error extracting images from {file_path}: {e}")
        return {"error": str(e), "images": []}


def list_images(settings: Settings, input_directory_str: str) -> List[Dict[str, Any]]:
    """List image files in the directory.

    Returns:
        List of image file information dictionaries
    """
    input_directory = Path(input_directory_str).resolve()

    logger.debug(f"Listing images in {input_directory}")
    images = []

    try:
        supported_formats = set(settings.image_processing.supported_formats)

        for file_path in input_directory.rglob("*"):
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')

                if extension in supported_formats:
                    # Security check
                    if not is_path_safe(settings, input_directory, file_path):
                        logger.warning(f"Skipping file outside base directory: {file_path}")
                        continue

                    # Check image size
                    if not is_file_size_within_limits(settings, file_path):
                        logger.warning(f"Skipping large image: {file_path}")
                        continue

                    images.append({
                        "path": str(file_path),
                        "filename": file_path.name,
                        "format": extension,
                        "size": file_path.stat().st_size
                    })

        logger.info(f"Found {len(images)} image files")
        return images
    except Exception as e:
        logger.exception(f"Error listing images: {e}")
        raise


def resize_image(image_path, max_dimension=768):
    """
    Resizes image to max_dimension and returns base64 string.
    768px is a sweet spot: high enough for detail, low enough for speed/context.
    """
    try:
        with Image.open(image_path) as img:
            # 1. Convert to RGB to handle PNGs with transparency or specialized modes
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')

            # 2. Get original dimensions
            width, height = img.size

            # 3. Calculate new size while preserving aspect ratio
            # Only resize if the image is larger than the limit
            if width > max_dimension or height > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            # 4. Save to buffer (in memory, not on disk)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)  # JPEG is efficient for LLMs

            return buffered.getvalue()

    except Exception as e:
        raise Exception(f"Error resizing image: {e}")


def describe_image(settings: Settings, image_path: str, prompts: Dict[str, Any],
                   language: str = "en-US", cache_results: bool = True) -> str:
    """Generate a description of an image using a vision-language model.
    
    Args:
        settings: The project settings
        image_path: Path to the image file
        prompts: Prompts configuration dictionary
        language: Target language for description
        cache_results: Whether to cache the description
        
    Returns:
        String description of the image
    """
    image_path = Path(image_path).resolve()
    logger.info(f"Describing image: {image_path}")

    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        raise Exception("Image file not found")

    # Check for cached description
    cache_path = get_cache_path(settings, image_path, prefix="img_desc_", ext="txt", language=language)
    if cache_results:
        cached_desc = read_cache(cache_path)
        if cached_desc is not None:
            # Cache normally stores JSON with a "description" field
            if isinstance(cached_desc, dict):
                return cached_desc.get("description", "")
            # Backward compatibility: handle legacy string-only cache entries
            if isinstance(cached_desc, str):
                return cached_desc
            logger.warning(f"Unexpected cache format for {cache_path}: {type(cached_desc).__name__}")

    try:
        filename = image_path.name

        # Get preprocessor prompts
        preprocessor_prompts = prompts.get('preprocessor', {})
        system_prompt_template = preprocessor_prompts.get('image_description_system_prompt', '')

        system_prompt = system_prompt_template.format(language=language)

        logger.info(f"Generating description for image: {filename}")

        try:
            # Get vision model from settings
            vision_llm = get_llm(
                settings,
                temperature=settings.vision_model.temperature,
                model=settings.vision_model.model,
                provider=settings.vision_model.provider
            )

            image_data = resize_image(image_path)
            b64_image_data = base64.b64encode(image_data).decode("utf-8")

            # Create message with image - separate system and user content
            message_content = [
                {
                    "type": "text",
                    "text": system_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image_data}"
                    }
                }
            ]

            message = HumanMessage(content=message_content)
            logger.info(f"Calling LLM with message={str(message)[:100]}...")
            response = vision_llm.invoke([message])
            logger.info(f"Response from LLM: {response}")
            description = response.content

            if not description:
                raise Exception("Got empty description")

            # Extract content from <result> tags if present
            if '<result>' in description and '</result>' in description:
                start = description.find('<result>') + 8
                end = description.find('</result>')
                description = description[start:end].strip()

        except Exception as e:
            logger.warning(f"Vision LLM description failed, using fallback: {e}")
            description = f"Image - {filename}"

        # Cache the description
        if cache_results:
            write_cache(cache_path, {"description": description})

        return description

    except Exception as e:
        logger.exception(f"Error describing image {image_path}: {e}")
        # Return basic description on error - use consistent format
        return f"Image from {image_path.parent.name}: {image_path.name}"
