"""Long-term memory management for AI Book Composer.

This module provides a disk-based storage system for file contents to avoid
context overflow. Full content is stored on disk while only summaries are kept
in the agent state (short-term memory).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Manages long-term storage of file contents to prevent context overflow.
    
    Stores full content on disk while keeping only summaries in memory.
    Provides on-demand retrieval of full content when needed.
    """
    
    def __init__(self, storage_dir: str):
        """Initialize long-term memory storage.
        
        Args:
            storage_dir: Directory to store long-term memory files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.content_file = self.storage_dir / "content_storage.json"
        self._content_cache: Dict[str, Dict[str, Any]] = {}
        self._load_storage()
    
    def _load_storage(self):
        """Load existing storage from disk."""
        if self.content_file.exists():
            try:
                with open(self.content_file, 'r', encoding='utf-8') as f:
                    self._content_cache = json.load(f)
                logger.info(f"Loaded {len(self._content_cache)} items from long-term memory")
            except Exception as e:
                logger.warning(f"Could not load long-term memory: {e}")
                self._content_cache = {}
    
    def _save_storage(self):
        """Save storage to disk."""
        try:
            with open(self.content_file, 'w', encoding='utf-8') as f:
                json.dump(self._content_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving long-term memory: {e}")
    
    def store_content(self, file_path: str, content_info: Dict[str, Any]) -> Dict[str, Any]:
        """Store full content in long-term memory and return summary version.
        
        Args:
            file_path: Path to the file
            content_info: Dictionary with keys: name, path, type, content, summary
        
        Returns:
            Dictionary with summary version (without full content)
        """
        # Store full content in long-term memory
        self._content_cache[file_path] = {
            'name': content_info.get('name', ''),
            'path': content_info.get('path', ''),
            'type': content_info.get('type', ''),
            'content': content_info.get('content', ''),
            'summary': content_info.get('summary', '')
        }
        self._save_storage()
        
        # Return summary version for short-term memory (without full content)
        return {
            'name': content_info.get('name', ''),
            'path': content_info.get('path', ''),
            'type': content_info.get('type', ''),
            'summary': content_info.get('summary', ''),
            'content_length': len(content_info.get('content', ''))
        }
    
    def retrieve_content(self, file_path: str, start_char: int = 0, length: Optional[int] = None) -> Optional[str]:
        """Retrieve full or partial content from long-term memory.
        
        Args:
            file_path: Path to the file
            start_char: Starting character position (default: 0)
            length: Number of characters to retrieve (None for all)
        
        Returns:
            Content string or None if not found
        """
        content_info = self._content_cache.get(file_path)
        if not content_info:
            logger.warning(f"Content not found in long-term memory: {file_path}")
            return None
        
        content = content_info.get('content', '')
        if length is None:
            return content[start_char:]
        else:
            return content[start_char:start_char + length]
    
    def get_content_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get content info (including full content) from long-term memory.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Content info dictionary or None if not found
        """
        return self._content_cache.get(file_path)
    
    def has_content(self, file_path: str) -> bool:
        """Check if content exists in long-term memory.
        
        Args:
            file_path: Path to the file
        
        Returns:
            True if content exists, False otherwise
        """
        return file_path in self._content_cache
    
    def clear(self):
        """Clear all stored content."""
        self._content_cache = {}
        if self.content_file.exists():
            self.content_file.unlink()
        logger.info("Long-term memory cleared")
    
    def get_summary(self, file_path: str) -> Optional[str]:
        """Get summary for a file from long-term memory.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Summary string or None if not found
        """
        content_info = self._content_cache.get(file_path)
        if content_info:
            return content_info.get('summary', '')
        return None
