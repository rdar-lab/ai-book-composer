"""Unit tests for long-term memory functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from ai_book_composer.long_term_memory import LongTermMemory


class TestLongTermMemory:
    """Test suite for LongTermMemory class."""

    def test_initialization(self):
        """Test that LongTermMemory initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            assert ltm.storage_dir == Path(tmpdir)
            assert ltm.storage_dir.exists()
            assert ltm.content_file == Path(tmpdir) / "content_storage.json"

    def test_store_and_retrieve_content(self):
        """Test storing and retrieving content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            # Store content
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'This is test content with some length',
                'summary': 'Test summary'
            }
            
            summary_version = ltm.store_content('/path/to/test.txt', content_info)
            
            # Check summary version doesn't have full content
            assert 'content' not in summary_version
            assert summary_version['summary'] == 'Test summary'
            assert summary_version['content_length'] == len('This is test content with some length')
            
            # Retrieve full content
            retrieved = ltm.retrieve_content('/path/to/test.txt')
            assert retrieved == 'This is test content with some length'

    def test_retrieve_partial_content(self):
        """Test retrieving partial content with start_char and length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                'summary': 'Alphabet'
            }
            
            ltm.store_content('/path/to/test.txt', content_info)
            
            # Retrieve partial content
            partial = ltm.retrieve_content('/path/to/test.txt', start_char=5, length=10)
            assert partial == 'FGHIJKLMNO'
            
            # Retrieve from start_char to end
            partial_to_end = ltm.retrieve_content('/path/to/test.txt', start_char=20)
            assert partial_to_end == 'UVWXYZ'

    def test_retrieve_nonexistent_content(self):
        """Test retrieving content that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            result = ltm.retrieve_content('/nonexistent/file.txt')
            assert result is None

    def test_has_content(self):
        """Test checking if content exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            assert not ltm.has_content('/path/to/test.txt')
            
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'Test content',
                'summary': 'Test'
            }
            ltm.store_content('/path/to/test.txt', content_info)
            
            assert ltm.has_content('/path/to/test.txt')
            assert not ltm.has_content('/other/file.txt')

    def test_get_summary(self):
        """Test getting summary for a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'Full content here',
                'summary': 'Short summary'
            }
            ltm.store_content('/path/to/test.txt', content_info)
            
            summary = ltm.get_summary('/path/to/test.txt')
            assert summary == 'Short summary'

    def test_clear(self):
        """Test clearing all stored content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            # Store some content
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'Test content',
                'summary': 'Test'
            }
            ltm.store_content('/path/to/test.txt', content_info)
            assert ltm.has_content('/path/to/test.txt')
            
            # Clear
            ltm.clear()
            assert not ltm.has_content('/path/to/test.txt')
            assert not ltm.content_file.exists()

    def test_persistence(self):
        """Test that content persists across LongTermMemory instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first instance and store content
            ltm1 = LongTermMemory(tmpdir)
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'Persistent content',
                'summary': 'Persistent'
            }
            ltm1.store_content('/path/to/test.txt', content_info)
            
            # Create second instance - should load existing data
            ltm2 = LongTermMemory(tmpdir)
            assert ltm2.has_content('/path/to/test.txt')
            assert ltm2.retrieve_content('/path/to/test.txt') == 'Persistent content'
            assert ltm2.get_summary('/path/to/test.txt') == 'Persistent'

    def test_get_content_info(self):
        """Test getting full content info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            content_info = {
                'name': 'test.txt',
                'path': '/path/to/test.txt',
                'type': 'text',
                'content': 'Test content',
                'summary': 'Test summary'
            }
            ltm.store_content('/path/to/test.txt', content_info)
            
            retrieved_info = ltm.get_content_info('/path/to/test.txt')
            assert retrieved_info['name'] == 'test.txt'
            assert retrieved_info['path'] == '/path/to/test.txt'
            assert retrieved_info['type'] == 'text'
            assert retrieved_info['content'] == 'Test content'
            assert retrieved_info['summary'] == 'Test summary'

    def test_multiple_files(self):
        """Test storing and retrieving multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(tmpdir)
            
            # Store multiple files
            for i in range(5):
                content_info = {
                    'name': f'test{i}.txt',
                    'path': f'/path/to/test{i}.txt',
                    'type': 'text',
                    'content': f'Content for file {i}',
                    'summary': f'Summary {i}'
                }
                ltm.store_content(f'/path/to/test{i}.txt', content_info)
            
            # Verify all files are stored
            for i in range(5):
                assert ltm.has_content(f'/path/to/test{i}.txt')
                assert ltm.retrieve_content(f'/path/to/test{i}.txt') == f'Content for file {i}'
                assert ltm.get_summary(f'/path/to/test{i}.txt') == f'Summary {i}'
