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
            print(f"         ❌ Madde bulunamadı")
            return []
        
        # Process each article
        for i in tqdm(range(len(matches)), desc="         Madde işleniyor", ncols=70):
            # Defensive bounds to avoid index errors
            try:
                if i >= len(matches):
                    break
                match_obj = matches[i]
                start = match_obj.start()
                # Safely get end position
                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(text)
            except (IndexError, AttributeError, Exception) as e:
                print(f"         ⚠️  Madde {i+1} atlanıyor: {e}")
                continue
            
            # Validate bounds
            if start < 0 or end <= start or start >= len(text) or end > len(text):
                continue
            
            try:
                article_text = text[start:end].strip()
            except Exception:
                continue
                
            if not article_text or len(article_text) < 10:
                continue
                
            # Safely extract article number
            article_num = re.search(r'MADDE\s+(\d+[A-Z]*)', article_text, re.IGNORECASE)
            try:
                article_number = article_num.group(1) if article_num and article_num.groups() else str(i + 1)
            except (AttributeError, IndexError):
                article_number = str(i + 1)
            
            # If article is too long, split it into smaller chunks
            if len(article_text) > self.chunk_size * 2:
                # Split long article into sub-chunks
                sub_chunks = self._split_long_article(article_text, article_number)
                if not sub_chunks:
                    continue
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
        if not article_text or len(article_text.strip()) < 10:
            return []
            
        try:
            paragraphs = [p for p in article_text.split('\n\n') if p and p.strip()]
        except Exception:
            paragraphs = [article_text]
            
        if not paragraphs:
            return []
            
        sub_chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if not para or not para.strip():
                continue
            para = para.strip()
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                trimmed = current_chunk.strip()
                if trimmed:
                    sub_chunks.append(trimmed)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk and current_chunk.strip():
            sub_chunks.append(current_chunk.strip())
        
        return [c for c in sub_chunks if c and len(c) > 10]
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Article-based chunking for legal documents (MADDE-based)
        """
        if not text:
            return []
        
        print(f"         ✂️  Chunk'lara bölünüyor...")
        text_length = len(text)
        print(f"         📏 Toplam karakter: {text_length:,}")
        
        # Use article-based chunking only
        if 'MADDE' in text.upper():
            return self.create_chunks_by_article(text, metadata)
        else:
            print(f"         ❌ Hukuki doküman değil, 'MADDE' bulunamadı")
            return []
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of text chunks with metadata
        """
        try:
            # Extract text
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            if not raw_text:
                print("         ⚠️  Metin çıkarılamadı")
                return []
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 10:
                print("         ⚠️  Temizlenmiş metin çok kısa")
                return []
            
            # Create metadata
            metadata = {
                'source': os.path.basename(pdf_path),
                'total_chars': len(cleaned_text)
            }
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text, metadata)
            
            if not chunks:
                print("         ❌ Hiç chunk oluşturulamadı")
                return []
            
            return chunks
        except Exception as e:
            print(f"         ❌ PDF işleme hatası: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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

