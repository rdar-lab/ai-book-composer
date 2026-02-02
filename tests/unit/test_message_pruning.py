"""Unit tests for message history pruning functionality."""
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from src.ai_book_composer.llm import ToolFixer


class TestMessagePruning:
    """Test suite for message history pruning."""

    def test_prune_history_keeps_system_prompt(self):
        """Test that system prompt is always kept."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            ToolMessage(content="Tool response", name="test_tool", tool_call_id="1"),
        ]

        pruned = ToolFixer._prune_history(messages)

        assert isinstance(pruned[0], SystemMessage)
        assert pruned[0].content == "You are a helpful assistant"

    def test_prune_history_compresses_old_tool_messages(self):
        """Test that old ToolMessages get compressed."""
        large_content = "x" * 10000  # 10KB of content

        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Request 1"),
            AIMessage(content="Response 1"),
            ToolMessage(content=large_content, name="get_file_content", tool_call_id="1"),
            HumanMessage(content="Request 2"),
            AIMessage(content="Response 2"),
            ToolMessage(content=large_content, name="get_file_content", tool_call_id="2"),
            HumanMessage(content="Request 3"),
            AIMessage(content="Response 3"),
            ToolMessage(content=large_content, name="get_file_content", tool_call_id="3"),
            HumanMessage(content="Request 4"),
            AIMessage(content="Response 4"),
        ]

        pruned = ToolFixer._prune_history(messages)

        # First tool message should be compressed (old) - keeps 200 char preview
        assert len(pruned[3].content) < 500  # Much smaller than original
        assert "truncated" in pruned[3].content.lower()
        assert "get_file_content" in pruned[3].content

        # Second tool message should be compressed (old)
        assert len(pruned[6].content) < 500
        assert "truncated" in pruned[6].content.lower()

    def test_prune_history_keeps_recent_messages(self):
        """Test that recent messages (last 4) are kept full."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Old request"),
            AIMessage(content="Old response"),
            ToolMessage(content="x" * 1000, name="tool1", tool_call_id="1"),  # Old - should compress
            HumanMessage(content="Recent request"),
            AIMessage(content="Recent response"),
            ToolMessage(content="Recent tool output", name="tool2", tool_call_id="2"),  # Recent - keep
            HumanMessage(content="Latest request"),
        ]

        pruned = ToolFixer._prune_history(messages)

        # Old tool message should be compressed - keeps 200 char preview
        assert len(pruned[3].content) < 500
        assert "truncated" in pruned[3].content.lower()

        # Recent tool message should be kept (within last 4)
        assert pruned[6].content == "Recent tool output"

    def test_prune_history_compresses_large_recent_tool_messages(self):
        """Test that even recent ToolMessages get compressed if they're very large (>3000 chars)."""
        huge_content = "y" * 10000  # 10KB
        old_huge_content = "z" * 10000  # 10KB for old message

        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Old request 1"),
            AIMessage(content="Old response 1"),
            ToolMessage(content=old_huge_content, name="get_file_content", tool_call_id="0"),  # Old and huge
            HumanMessage(content="Request 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Request 3"),
            AIMessage(content="Response 3"),
            HumanMessage(content="Request 4"),
            AIMessage(content="Response 4"),
            ToolMessage(content=huge_content, name="get_file_content", tool_call_id="1"),  # Recent and huge
        ]

        pruned = ToolFixer._prune_history(messages)

        # Old huge message should be compressed to 200 chars
        assert len(pruned[3].content) < 500
        assert "truncated" in pruned[3].content.lower()

        # Recent huge message should be compressed but keep more (1000 chars)
        assert len(pruned[10].content) < 1500  # Smaller than original but more than old messages
        assert "truncated" in pruned[10].content.lower()

        # Verify recent messages keep more content than old messages
        assert len(pruned[10].content) > len(pruned[3].content)

    def test_prune_history_trims_large_user_prompts(self):
        """Test that large user prompts in old messages are NOT trimmed anymore."""
        large_prompt = "x" * 5000

        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content=large_prompt),  # Old and large - but DON'T trim
            AIMessage(content="Response"),
            HumanMessage(content="Recent request"),
            AIMessage(content="Recent response"),
            ToolMessage(content="tool output", name="tool", tool_call_id="1"),
            HumanMessage(content="Latest"),
        ]

        pruned = ToolFixer._prune_history(messages)

        # HumanMessages should NOT be trimmed - they're important for generation
        assert pruned[1].content == large_prompt

        # Recent prompts should be kept
        assert pruned[3].content == "Recent request"
        assert pruned[6].content == "Latest"

    def test_prune_history_handles_empty_messages(self):
        """Test that pruning handles empty message list."""
        messages = []
        pruned = ToolFixer._prune_history(messages)
        assert pruned == []

    def test_prune_history_handles_single_message(self):
        """Test that pruning handles single message."""
        messages = [SystemMessage(content="Only message")]
        pruned = ToolFixer._prune_history(messages)
        assert len(pruned) == 1
        assert pruned[0].content == "Only message"

    def test_prune_history_preserves_message_types(self):
        """Test that pruning preserves message types."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="Human"),
            AIMessage(content="AI"),
            ToolMessage(content="Tool", name="tool", tool_call_id="1"),
        ]

        pruned = ToolFixer._prune_history(messages)

        assert isinstance(pruned[0], SystemMessage)
        assert isinstance(pruned[1], HumanMessage)
        assert isinstance(pruned[2], AIMessage)
        assert isinstance(pruned[3], ToolMessage)

    def test_prune_history_deep_copies_messages(self):
        """Test that pruning doesn't modify original messages."""
        large_content = "x" * 10000
        original_tool_msg = ToolMessage(content=large_content, name="tool", tool_call_id="1")

        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Old request"),
            AIMessage(content="Old response"),
            original_tool_msg,
            HumanMessage(content="Recent request"),
            AIMessage(content="Recent response"),
        ]

        pruned = ToolFixer._prune_history(messages)

        # Original should not be modified
        assert len(original_tool_msg.content) == 10000
        # Pruned should be compressed - keeps 1000 chars for recent messages
        assert len(pruned[3].content) < 1500  # Compressed but more than 200 since it's recent

    def test_prune_history_with_many_tool_calls(self):
        """Test pruning with many tool calls simulating file access pattern."""
        # Expected compression ratio - at least 70% reduction
        messages: list[Any] = [SystemMessage(content="System prompt")]

        # Simulate 10 file access calls
        for i in range(10):
            messages.extend([
                HumanMessage(content=f"Get file {i}"),
                AIMessage(content=f"Getting file {i}"),
                ToolMessage(content="x" * 5000, name="get_file_content", tool_call_id=str(i)),
            ])

        pruned = ToolFixer._prune_history(messages)

        # Verify total size is much smaller than original
        original_size = sum(len(str(m.content)) for m in messages)
        pruned_size = sum(len(str(m.content)) for m in pruned)

        assert pruned_size < original_size * 0.3  # At least 70% reduction

        # Verify old tool messages are compressed - keeps 200 char preview
        # Messages[3] is first tool response (old)
        assert "truncated" in pruned[3].content.lower()
        # Last few should be less compressed or kept
        assert len(pruned) == len(messages)  # Same number of messages
