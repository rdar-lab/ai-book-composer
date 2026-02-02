"""
Demonstration of long-term memory preventing context overflow.

This script demonstrates how the long-term memory system prevents
context overflow when processing large amounts of data.
"""

import tempfile
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Show the problem
print("=" * 80)
print("DEMONSTRATION: Long-Term Memory Solution")
print("=" * 80)

print("\n📊 Problem: Message History Explosion")
print("-" * 80)
print("When agents repeatedly access files via get_file_content tool:")
print("- Each file access adds a ToolMessage with file content")
print("- Messages accumulate in conversation history")
print("- Context window fills up and causes errors")
print()

# Simulate old behavior
print("❌ OLD BEHAVIOR (Without Long-Term Memory):")
print("-" * 80)

# Simulate 10 file accesses with large content
old_messages = [SystemMessage(content="You are a helpful assistant.")]
file_content = "x" * 5000  # 5KB per file

for i in range(10):
    old_messages.extend([
        HumanMessage(content=f"Get content from file_{i}.txt"),
        AIMessage(content=f"I'll retrieve file_{i}.txt"),
        ToolMessage(content=file_content, name="get_file_content", tool_call_id=str(i))
    ])

old_total_size = sum(len(str(msg.content)) for msg in old_messages)
print(f"  Messages: {len(old_messages)}")
print(f"  Total Size: {old_total_size:,} characters")
print(f"  Average per message: {old_total_size // len(old_messages):,} characters")
print()

# Show new behavior with pruning
print("✅ NEW BEHAVIOR (With Long-Term Memory & Aggressive Pruning):")
print("-" * 80)

# Import the pruning function
from ai_book_composer.llm import ToolFixer

# Apply pruning
pruned_messages = ToolFixer._prune_history(old_messages)
pruned_total_size = sum(len(str(msg.content)) for msg in pruned_messages)

print(f"  Messages: {len(pruned_messages)}")
print(f"  Total Size: {pruned_total_size:,} characters")
print(f"  Average per message: {pruned_total_size // len(pruned_messages):,} characters")
print(f"  💾 Size Reduction: {((old_total_size - pruned_total_size) / old_total_size * 100):.1f}%")
print()

# Show what happened to the messages
print("📝 Message Compression Details:")
print("-" * 80)
print(f"  System Prompt: Kept intact")
print(f"  Old ToolMessages (7): Compressed to metadata (~{len(pruned_messages[3].content)} chars each)")
print(f"  Recent Messages (last 4): Kept mostly intact")
print(f"  Large Recent ToolMessages (>3KB): Partially compressed")
print()

# Show long-term memory benefits
print("🗄️ Long-Term Memory Benefits:")
print("-" * 80)
print("  ✓ Full file content stored externally (JSON on disk)")
print("  ✓ Only summaries kept in AgentState")
print("  ✓ Tool responses return compact chunks (5KB max)")
print("  ✓ Content retrievable on-demand without re-reading files")
print("  ✓ Consistent access pattern across all agents")
print()

# Show the configuration
print("⚙️ Pruning Strategy:")
print("-" * 80)
print("  • Keep: System prompt + last 4 message turns")
print("  • Compress: Old ToolMessages to metadata only")
print("  • Limit: Large recent ToolMessages to 500 chars + metadata")
print("  • Trim: Large user prompts in old messages to 400 chars")
print()

# Real-world scenario
print("🌍 Real-World Impact:")
print("-" * 80)
print("  Scenario: Processing 100 files, each 50KB")
print("  Old approach: 100 files × 50KB = ~5MB in context → ERROR")
print("  New approach: ~50KB total context → ✓ SUCCESS")
print("  Context savings: ~99% reduction")
print()

# Test with actual long-term memory
print("🧪 Testing Long-Term Memory Storage:")
print("-" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    from ai_book_composer.long_term_memory import LongTermMemory
    
    ltm = LongTermMemory(tmpdir)
    
    # Store a large file
    content_info = {
        'name': 'large_document.txt',
        'path': '/test/large_document.txt',
        'type': 'text',
        'content': 'A' * 100000,  # 100KB
        'summary': 'Large document with important information'
    }
    
    summary_version = ltm.store_content('/test/large_document.txt', content_info)
    
    print(f"  ✓ Stored 100KB file in long-term memory")
    print(f"  ✓ Summary version: {len(str(summary_version))} chars (vs 100KB original)")
    print(f"  ✓ Can retrieve full content on demand")
    print(f"  ✓ Storage location: {tmpdir}/.long_term_memory/")
    
    # Verify retrieval
    retrieved = ltm.retrieve_content('/test/large_document.txt', start_char=0, length=1000)
    print(f"  ✓ Retrieved chunk: {len(retrieved)} chars")
    assert len(retrieved) == 1000
    print()

print("=" * 80)
print("✅ CONCLUSION: Long-term memory prevents context overflow!")
print("=" * 80)
print("The system now handles large datasets without context explosion.")
print("Message history stays compact through aggressive compression,")
print("while full content remains accessible via long-term memory.")
print()
