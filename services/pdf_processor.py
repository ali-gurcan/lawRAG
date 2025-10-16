"""
PDF Processing Module for RAG System
Handles PDF reading, chunking, and text extraction
"""
import os
import re
from typing import List, Dict
import PyPDF2
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Number of characters per chunk (increased for speed)
            chunk_overlap: Overlap between chunks for context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"         📖 Toplam sayfa: {total_pages}")
                
                for page_num in tqdm(range(total_pages), desc="         Sayfalar çıkarılıyor", ncols=70):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                print(f"         ✅ {len(text):,} karakter çıkarıldı")
                return text
        
        except Exception as e:
            print(f"         ❌ Hata: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        print(f"         🧹 Metin temizleniyor...")
        original_length = len(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Turkish characters
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}\"\'çÇğĞıİöÖşŞüÜ-]', '', text)
        
        # Remove page numbers (optional)
        text = re.sub(r'\b\d+\b\s*$', '', text, flags=re.MULTILINE)
        
        cleaned_text = text.strip()
        cleaned_length = len(cleaned_text)
        reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        
        print(f"         ✅ Temizleme tamamlandı ({reduction_pct:.1f}% azaldı)")
        
        return cleaned_text
    
    def create_chunks_by_article(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text by legal articles (MADDE)
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
        
        print(f"         ✂️  Maddelere göre chunk'lara bölünüyor...")
        
        # Find all article boundaries (MADDE X-)
        import re
        article_pattern = r'MADDE\s+\d+[A-Z]*\s*-'
        matches = list(re.finditer(article_pattern, text, re.IGNORECASE))
        
        print(f"         📋 {len(matches)} madde bulundu")
        
        chunks = []
        chunk_id = 0
        
        from tqdm import tqdm
        
        if not matches:
            # Fallback to sentence-based chunking if no articles found
            print(f"         ⚠️  Madde bulunamadı, normal chunking kullanılıyor...")
            return self.create_chunks_simple(text, metadata)
        
        # Process each article
        for i in tqdm(range(len(matches)), desc="         Madde işleniyor", ncols=70):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            article_text = text[start:end].strip()
            article_num = re.search(r'MADDE\s+(\d+[A-Z]*)', article_text, re.IGNORECASE)
            article_number = article_num.group(1) if article_num else str(i + 1)
            
            # If article is too long, split it into smaller chunks
            if len(article_text) > self.chunk_size * 2:
                # Split long article into sub-chunks
                sub_chunks = self._split_long_article(article_text, article_number)
                for sub_chunk in sub_chunks:
                    chunk = {
                        'id': chunk_id,
                        'text': sub_chunk,
                        'start_char': start,
                        'end_char': end,
                        'metadata': {
                            **(metadata or {}),
                            'article': f'MADDE {article_number}',
                            'type': 'article'
                        }
                    }
                    chunks.append(chunk)
                    chunk_id += 1
            else:
                # Keep article as single chunk
                chunk = {
                    'id': chunk_id,
                    'text': article_text,
                    'start_char': start,
                    'end_char': end,
                    'metadata': {
                        **(metadata or {}),
                        'article': f'MADDE {article_number}',
                        'type': 'article'
                    }
                }
                chunks.append(chunk)
                chunk_id += 1
        
        print(f"         ✅ {len(chunks)} chunk oluşturuldu")
        avg_size = sum(len(c['text']) for c in chunks) // len(chunks) if chunks else 0
        print(f"         📊 Ortalama chunk boyutu: {avg_size:,} karakter")
        
        return chunks
    
    def _split_long_article(self, article_text: str, article_number: str) -> List[str]:
        """Split a long article into smaller sub-chunks at paragraph boundaries"""
        paragraphs = article_text.split('\n\n')
        sub_chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                sub_chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def create_chunks_simple(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Simple sentence-based chunking (fallback method)
        """
        chunks = []
        chunk_id = 0
        text_length = len(text)
        start = 0
        
        from tqdm import tqdm
        
        with tqdm(total=text_length, desc="         Chunk progress", 
                  unit="char", unit_scale=True, ncols=70, mininterval=0.5) as pbar:
            
            while start < text_length:
                old_start = start
                end = min(start + self.chunk_size, text_length)
                
                # Try to break at sentence boundary
                if end < text_length:
                    search_start = start + int(self.chunk_size * 0.8)
                    for delimiter in ['. ', '! ', '? ', '\n\n']:
                        last_delimiter = text.rfind(delimiter, search_start, end)
                        if last_delimiter != -1:
                            end = last_delimiter + len(delimiter)
                            break
                
                chunk_text = text[start:end].strip()
                
                if chunk_text and len(chunk_text) > 50:  # Skip very small chunks
                    chunk = {
                        'id': chunk_id,
                        'text': chunk_text,
                        'start_char': start,
                        'end_char': end,
                        'metadata': metadata or {}
                    }
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Calculate next start with overlap
                next_start = end - self.chunk_overlap
                if next_start <= start:
                    next_start = start + max(100, self.chunk_size // 2)
                
                pbar.update(next_start - old_start)
                start = next_start
        
        return chunks
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Smart chunking: Uses article-based chunking for legal documents,
        falls back to sentence-based chunking otherwise
        """
        if not text:
            return []
        
        print(f"         ✂️  Chunk'lara bölünüyor...")
        text_length = len(text)
        print(f"         📏 Toplam karakter: {text_length:,}")
        
        # Try article-based chunking first
        if 'MADDE' in text.upper():
            return self.create_chunks_by_article(text, metadata)
        else:
            print(f"         ⚠️  Hukuki doküman değil, genel chunking kullanılıyor...")
            return self.create_chunks_simple(text, metadata)
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of text chunks with metadata
        """
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text:
            print("         ⚠️  Metin çıkarılamadı")
            return []
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Create metadata
        metadata = {
            'source': os.path.basename(pdf_path),
            'total_chars': len(cleaned_text)
        }
        
        # Create chunks
        chunks = self.create_chunks(cleaned_text, metadata)
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
        
        Returns:
            Combined list of chunks from all PDFs
        """
        all_chunks = []
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        print(f"\n📚 Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        print(f"\n✅ Total chunks from all PDFs: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # Test the processor
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    
    # Process Anayasa PDF
    pdf_path = "docs/anayasa.pdf"
    if os.path.exists(pdf_path):
        chunks = processor.process_pdf(pdf_path)
        
        # Show sample chunks
        print("\n📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Text: {chunk['text'][:200]}...")
            print(f"Length: {len(chunk['text'])} chars")
    else:
        print(f"❌ PDF not found: {pdf_path}")

