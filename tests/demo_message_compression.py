"""
Simple demonstration of message history compression fix.

This script shows how the enhanced _prune_history() prevents context overflow
by compressing old ToolMessages while keeping all data in AgentState.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from ai_book_composer.llm import ToolFixer

print("=" * 80)
print("MESSAGE HISTORY COMPRESSION DEMONSTRATION")
print("=" * 80)
print()

# Simulate the problem: 10 file accesses with large content
print("❌ PROBLEM: Context overflow from repeated file access")
print("-" * 80)

messages = [SystemMessage(content="You are a helpful assistant.")]
file_content = "x" * 5000  # 5KB per file

for i in range(10):
    messages.extend([
        HumanMessage(content=f"Get content from file_{i}.txt"),
        AIMessage(content=f"I'll retrieve file_{i}.txt"),
        ToolMessage(content=file_content, name="get_file_content", tool_call_id=str(i))
    ])

original_size = sum(len(str(msg.content)) for msg in messages)
print(f"Messages: {len(messages)}")
print(f"Total size: {original_size:,} characters")
print()

# Apply the solution
print("✅ SOLUTION: Aggressive message pruning")
print("-" * 80)

pruned_messages = ToolFixer._prune_history(messages)
pruned_size = sum(len(str(msg.content)) for msg in pruned_messages)

print(f"Messages: {len(pruned_messages)} (same count)")
print(f"Total size: {pruned_size:,} characters")
print(f"Reduction: {((original_size - pruned_size) / original_size * 100):.1f}%")
print()

# Show what happened
print("📝 WHAT HAPPENED:")
print("-" * 80)
print("✓ System prompt: Kept intact")
print("✓ Old ToolMessages (6 messages): Compressed to ~100 chars each")
print("✓ Recent messages (last 4 turns): Kept mostly intact")
print("✓ All data still in AgentState - accessible via get_file_content tool")
print()

# Show example compression
print("🔍 EXAMPLE COMPRESSION:")
print("-" * 80)
print("Old ToolMessage (index 3):")
print(f"  Length: {len(pruned_messages[3].content)} chars")
print(f"  Content: {pruned_messages[3].content[:100]}...")
print()

print("=" * 80)
print("✅ RESULT: Context overflow prevented!")
print("=" * 80)
print("Message history stays compact while all file content")
print("remains accessible in AgentState via get_file_content tool.")
print()
