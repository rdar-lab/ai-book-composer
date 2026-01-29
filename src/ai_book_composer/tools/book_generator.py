"""Tool for generating final book in RTF format."""

from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class BookGeneratorTool:
    """Tool to generate final book in RTF format."""
    
    name = "generate_final_book"
    description = "Generate the final book in RTF format"
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        title: str,
        author: str,
        chapters: List[Dict[str, str]],
        references: List[str],
        output_filename: str = "book.rtf"
    ) -> Dict[str, Any]:
        """Generate final book in RTF format.
        
        Args:
            title: Book title
            author: Book author
            chapters: List of chapter dictionaries with 'title' and 'content'
            references: List of reference strings
            output_filename: Output filename
            
        Returns:
            Dictionary with file path and status
        """
        try:
            file_path = self.output_directory / output_filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                # RTF header
                f.write(r"{\rtf1\ansi\deff0" + "\n")
                f.write(r"{\fonttbl{\f0\froman Times New Roman;}{\f1\fswiss Arial;}}" + "\n")
                f.write(r"{\colortbl;\red0\green0\blue0;\red0\green0\blue255;}" + "\n")
                
                # Title page
                f.write(r"\pard\qc\f1\fs48\b " + self._escape_rtf(title) + r"\par" + "\n")
                f.write(r"\fs24 " + self._escape_rtf(author) + r"\par" + "\n")
                f.write(r"\fs20 " + datetime.now().strftime("%B %Y") + r"\par" + "\n")
                f.write(r"\page" + "\n")
                
                # Table of contents
                f.write(r"\pard\f1\fs32\b Table of Contents\par" + "\n")
                f.write(r"\pard\f0\fs20\b0" + "\n")
                for i, chapter in enumerate(chapters, 1):
                    chapter_title = chapter.get("title", f"Chapter {i}")
                    f.write(f"Chapter {i}: {self._escape_rtf(chapter_title)}\\par\n")
                f.write(r"\page" + "\n")
                
                # Chapters
                for i, chapter in enumerate(chapters, 1):
                    chapter_title = chapter.get("title", f"Chapter {i}")
                    chapter_content = chapter.get("content", "")
                    
                    f.write(r"\pard\f1\fs32\b Chapter " + str(i) + ": " + 
                           self._escape_rtf(chapter_title) + r"\par" + "\n")
                    f.write(r"\pard\f0\fs24\b0" + "\n")
                    
                    # Process content paragraphs
                    paragraphs = chapter_content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            f.write(self._escape_rtf(para.strip()) + r"\par\par" + "\n")
                    
                    f.write(r"\page" + "\n")
                
                # References
                if references:
                    f.write(r"\pard\f1\fs32\b References\par" + "\n")
                    f.write(r"\pard\f0\fs20\b0" + "\n")
                    for ref in references:
                        f.write(self._escape_rtf(ref) + r"\par" + "\n")
                
                # RTF footer
                f.write("}\n")
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chapter_count": len(chapters),
                "reference_count": len(references)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _escape_rtf(self, text: str) -> str:
        """Escape special RTF characters.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        # Replace special characters
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{")
        text = text.replace("}", "\\}")
        text = text.replace("\n", " ")
        return text
