#!/usr/bin/env python3
"""Example script to run AI Book Composer with sample data."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_book_composer import BookComposerWorkflow


def main():
    """Run the book composer with example data."""
    
    # Paths
    input_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    output_dir = Path(__file__).parent / "sample-output"
    
    print("=" * 60)
    print("AI Book Composer - Example Run")
    print("=" * 60)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Check if input directory exists and has files
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    files = list(input_dir.glob("*"))
    if not files:
        print(f"Error: No files found in input directory: {input_dir}")
        return
    
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {f.name}")
    print()
    
    # Create workflow
    print("Initializing workflow...")
    workflow = BookComposerWorkflow(
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        language="en-US",
        book_title="Introduction to Artificial Intelligence and Machine Learning",
        book_author="AI Book Composer",
        max_iterations=2
    )
    
    # Run workflow
    print("Starting book composition...")
    print("This may take a few minutes depending on the content and LLM provider...")
    print()
    
    try:
        final_state = workflow.run()
        
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Status: {final_state.get('status')}")
        print(f"Chapters: {len(final_state.get('chapters', []))}")
        print(f"References: {len(final_state.get('references', []))}")
        print(f"Iterations: {final_state.get('iterations', 0)}")
        
        quality_score = final_state.get('quality_score')
        if quality_score is not None:
            print(f"Quality Score: {quality_score:.2f}")
        
        output_path = final_state.get('final_output_path')
        if output_path:
            print(f"Output File: {output_path}")
        
        feedback = final_state.get('critic_feedback')
        if feedback:
            print()
            print("Critic Feedback:")
            print("-" * 60)
            print(feedback)
        
        print()
        print("✓ Book composition completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
