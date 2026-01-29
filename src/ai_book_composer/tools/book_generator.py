"""Tool for generating final book in RTF format using PyRTF3."""

from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from PyRTF import *

from ..config import settings
from ..logging_config import logger


class BookGeneratorTool:
    """Tool to generate final book in RTF format."""
    
    name = "generate_final_book"
    description = "Generate the final book in RTF format"
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"BookGeneratorTool initialized for directory: {self.output_directory}")
    
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
        logger.info(f"Generating book: {title} by {author}")
        logger.debug(f"Chapters: {len(chapters)}, References: {len(references)}")
        
        try:
            file_path = self.output_directory / output_filename
            
            # Create document
            doc = Document()
            ss = doc.StyleSheet
            
            # Define styles
            title_style = TextStyle(TextPropertySet(font=ss.Fonts.Arial, size=48, bold=True))
            heading1_style = TextStyle(TextPropertySet(font=ss.Fonts.Arial, size=32, bold=True))
            heading2_style = TextStyle(TextPropertySet(font=ss.Fonts.Arial, size=24, bold=True))
            normal_style = TextStyle(TextPropertySet(font=ss.Fonts.TimesNewRoman, size=24))
            
            section = Section()
            doc.Sections.append(section)
            
            # Title page
            p_title = Paragraph(ss.ParagraphStyles.Heading1)
            p_title.append(title, title_style)
            section.append(p_title)
            
            p_author = Paragraph(ss.ParagraphStyles.Normal)
            p_author.append(f"By {author}", heading2_style)
            section.append(p_author)
            
            p_date = Paragraph(ss.ParagraphStyles.Normal)
            p_date.append(datetime.now().strftime("%B %Y"), normal_style)
            section.append(p_date)
            
            # Page break
            section.append(PAGE_BREAK)
            
            # Table of Contents
            toc_heading = Paragraph(ss.ParagraphStyles.Heading1)
            toc_heading.append("Table of Contents", heading1_style)
            section.append(toc_heading)
            
            for i, chapter in enumerate(chapters, 1):
                chapter_title = chapter.get("title", f"Chapter {i}")
                toc_entry = Paragraph(ss.ParagraphStyles.Normal)
                toc_entry.append(f"Chapter {i}: {chapter_title}", normal_style)
                section.append(toc_entry)
            
            section.append(PAGE_BREAK)
            
            # Chapters
            for i, chapter in enumerate(chapters, 1):
                chapter_title = chapter.get("title", f"Chapter {i}")
                chapter_content = chapter.get("content", "")
                
                # Chapter heading
                ch_heading = Paragraph(ss.ParagraphStyles.Heading1)
                ch_heading.append(f"Chapter {i}: {chapter_title}", heading1_style)
                section.append(ch_heading)
                
                # Chapter content - split into paragraphs
                paragraphs = chapter_content.split('\n\n')
                for para_text in paragraphs:
                    if para_text.strip():
                        para = Paragraph(ss.ParagraphStyles.Normal)
                        para.append(para_text.strip(), normal_style)
                        section.append(para)
                
                section.append(PAGE_BREAK)
            
            # References
            if references:
                ref_heading = Paragraph(ss.ParagraphStyles.Heading1)
                ref_heading.append("References", heading1_style)
                section.append(ref_heading)
                
                for ref in references:
                    ref_para = Paragraph(ss.ParagraphStyles.Normal)
                    ref_para.append(ref, normal_style)
                    section.append(ref_para)
            
            # Write to file
            renderer = Renderer()
            with open(file_path, 'w', encoding='utf-8') as f:
                renderer.Write(doc, f)
            
            logger.info(f"Book generated successfully: {file_path}")
            
            return {
                "success": True,
                "file_path": str(file_path),
                "chapter_count": len(chapters),
                "reference_count": len(references)
            }
        except Exception as e:
            logger.error(f"Error generating book: {e}")
            return {
                "success": False,
                "error": str(e)
            }
