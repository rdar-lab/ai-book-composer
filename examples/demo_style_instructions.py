#!/usr/bin/env python3
"""
Demonstration script for the style instructions feature.

This script shows how to use style instructions in different ways:
1. Via configuration file
2. Via command-line argument
3. Via Python API
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai_book_composer.config import Settings
from ai_book_composer.agents.state import create_initial_state

def demo_config_file():
    """Demonstrate style instructions via config file."""
    print("=" * 70)
    print("DEMO 1: Style Instructions via Configuration File")
    print("=" * 70)
    print()
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'book': {
                'output_language': 'en-US',
                'default_title': 'My Research Compilation',
                'default_author': 'Dr. Smith',
                'style_instructions': 'I want an academic book with formal language and in-depth analysis'
            }
        }
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    # Load settings from config file
    settings = Settings(temp_config_path)
    
    print(f"Config file: {temp_config_path}")
    print(f"Book title: {settings.book.default_title}")
    print(f"Book author: {settings.book.default_author}")
    print(f"Style instructions: \"{settings.book.style_instructions}\"")
    print()
    print("✓ The AI will generate chapters with formal, academic language")
    print()
    
    # Clean up
    Path(temp_config_path).unlink()


def demo_python_api():
    """Demonstrate style instructions via Python API."""
    print("=" * 70)
    print("DEMO 2: Style Instructions via Python API")
    print("=" * 70)
    print()
    
    # Example 1: Academic book
    print("Example 1: Academic Book")
    print("-" * 40)
    state = create_initial_state(
        input_directory='/tmp/research_papers',
        output_directory='/tmp/output',
        language='en-US',
        book_title='Advanced Machine Learning',
        book_author='Dr. Jane Doe',
        style_instructions='I want an academic book with formal language and technical depth'
    )
    print(f"Book: {state['book_title']}")
    print(f"Style: \"{state['style_instructions']}\"")
    print("→ Result: Formal, technical content with citations and detailed explanations")
    print()
    
    # Example 2: Light reading
    print("Example 2: Light Reading")
    print("-" * 40)
    state = create_initial_state(
        input_directory='/tmp/blog_posts',
        output_directory='/tmp/output',
        language='en-US',
        book_title='Simple Tech Explained',
        book_author='John Smith',
        style_instructions='I want it to be light reading with a conversational tone'
    )
    print(f"Book: {state['book_title']}")
    print(f"Style: \"{state['style_instructions']}\"")
    print("→ Result: Casual, easy-to-read content with simple explanations")
    print()
    
    # Example 3: Professional
    print("Example 3: Professional Reading")
    print("-" * 40)
    state = create_initial_state(
        input_directory='/tmp/business_docs',
        output_directory='/tmp/output',
        language='en-US',
        book_title='Business Strategy Guide',
        book_author='Corporate Authors',
        style_instructions='I want it to be professional reading material for executives'
    )
    print(f"Book: {state['book_title']}")
    print(f"Style: \"{state['style_instructions']}\"")
    print("→ Result: Polished, business-appropriate content with actionable insights")
    print()


def demo_cli_usage():
    """Show CLI usage examples."""
    print("=" * 70)
    print("DEMO 3: Style Instructions via CLI")
    print("=" * 70)
    print()
    
    examples = [
        {
            "title": "Academic Book",
            "command": """ai-book-composer \\
  -i ./research-papers \\
  -o ./output \\
  -t "Research Compilation" \\
  -a "Dr. Smith" \\
  --style-instructions "I want an academic book with formal language"
"""
        },
        {
            "title": "Light Reading",
            "command": """ai-book-composer \\
  -i ./blog-posts \\
  -o ./output \\
  -t "Easy Tech Guide" \\
  -a "John Doe" \\
  -s "I want it to be light reading for beginners"
"""
        },
        {
            "title": "Kids' Book",
            "command": """ai-book-composer \\
  -i ./stories \\
  -o ./output \\
  -t "Fun Adventures" \\
  -a "Children's Author" \\
  -s "I want it to be kids/fun reading material"
"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['title']}")
        print("-" * 40)
        print(example['command'])
        print()


def main():
    """Run all demonstrations."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "STYLE INSTRUCTIONS FEATURE DEMO" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    demo_config_file()
    input("Press Enter to continue to next demo...")
    print()
    
    demo_python_api()
    input("Press Enter to continue to next demo...")
    print()
    
    demo_cli_usage()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Style instructions can be provided in three ways:")
    print("  1. Configuration file (book.style_instructions)")
    print("  2. Command-line argument (--style-instructions or -s)")
    print("  3. Python API (style_instructions parameter)")
    print()
    print("Common use cases:")
    print("  • Academic books: Formal language, technical depth")
    print("  • Light reading: Conversational tone, simple explanations")
    print("  • Professional material: Business-appropriate, polished")
    print("  • Kids' content: Fun, engaging, age-appropriate")
    print()
    print("✓ The feature is fully integrated into the AI Book Composer!")
    print()


if __name__ == '__main__':
    main()
