import os
import json
import numpy as np

# Try importing PDF reader and embedding libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

CACHE_FILE = "rag_cache.json"
PDF_PATH = "Stock_Market_Master_Guide_150_Pages.pdf"

class RAGEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.embedder = None
        
        # Load embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use CPU-only and lightweight model
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Error loading SentenceTransformer: {e}")
                self.embedder = None
                
        self.load_or_build_index()

    def extract_pdf_chunks(self, path):
        """Extracts text page-by-page and splits it into chunks of ~600 characters."""
        chunks = []
        if not PDFPLUMBER_AVAILABLE:
            print("pdfplumber not available. Cannot parse PDF.")
            return chunks
            
        try:
            with pdfplumber.open(path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Split into paragraphs or size-based chunks
                    page_num = idx + 1
                    paragraphs = text.split("\n\n")
                    for p in paragraphs:
                        p = p.strip()
                        if len(p) < 40:
                            continue
                        
                        # If paragraph is too long, break it up
                        if len(p) > 1000:
                            words = p.split()
                            sub_chunks = []
                            current_chunk = []
                            current_len = 0
                            for word in words:
                                current_chunk.append(word)
                                current_len += len(word) + 1
                                if current_len > 600:
                                    sub_chunks.append(" ".join(current_chunk))
                                    current_chunk = []
                                    current_len = 0
                            if current_chunk:
                                sub_chunks.append(" ".join(current_chunk))
                            
                            for sc in sub_chunks:
                                chunks.append({
                                    "text": sc,
                                    "page": page_num,
                                    "source": "Stock Market Master Guide"
                                })
                        else:
                            chunks.append({
                                "text": p,
                                "page": page_num,
                                "source": "Stock Market Master Guide"
                            })
        except Exception as e:
            print(f"Failed to read PDF: {e}")
            
        return chunks

    def load_or_build_index(self):
        """Loads index from cache file or parses PDF and generates embeddings."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.chunks = data["chunks"]
                    if "embeddings" in data and self.embedder is not None:
                        self.embeddings = np.array(data["embeddings"])
                        print(f"Loaded {len(self.chunks)} chunks from RAG cache.")
                        return
            except Exception as e:
                print(f"Failed loading cache: {e}")

        # Build index
        if os.path.exists(PDF_PATH):
            print("Indexing PDF... This runs once and caches.")
            self.chunks = self.extract_pdf_chunks(PDF_PATH)
            
            if self.chunks and self.embedder is not None:
                try:
                    texts = [c["text"] for c in self.chunks]
                    emb_list = self.embedder.encode(texts, show_progress_bar=False)
                    self.embeddings = np.array(emb_list)
                    
                    # Save to cache
                    with open(CACHE_FILE, 'w') as f:
                        json.dump({
                            "chunks": self.chunks,
                            "embeddings": self.embeddings.tolist()
                        }, f)
                    print("RAG index built and cached successfully.")
                except Exception as e:
                    print(f"Embedding generation failed: {e}")
            elif self.chunks:
                # Cache raw chunks if embedding fails
                try:
                    with open(CACHE_FILE, 'w') as f:
                        json.dump({"chunks": self.chunks}, f)
                    print("Cached raw text chunks (Keyword search fallback active).")
                except Exception as e:
                    print(f"Failed saving raw chunks: {e}")
        else:
            print(f"PDF not found at {PDF_PATH}. RAG search will fall back to empty search.")

    def query(self, query_text, k=4):
        """Performs search using semantic embeddings or word-match fallback."""
        if not self.chunks:
            return []

        # Try semantic search first
        if self.embedder is not None and self.embeddings is not None:
            try:
                query_emb = self.embedder.encode(query_text, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, torch.tensor(self.embeddings, dtype=torch.float32))[0]
                top_results = torch.topk(cos_scores, k=min(k, len(self.chunks)))
                
                results = []
                for score, idx in zip(top_results[0], top_results[1]):
                    item = self.chunks[int(idx)].copy()
                    item["score"] = float(score)
                    results.append(item)
                return results
            except Exception as e:
                print(f"Semantic query error: {e}")

        # Keyword Match Fallback
        query_words = set(query_text.lower().split())
        scores = []
        for c in self.chunks:
            text_lower = c["text"].lower()
            match_count = sum(1 for w in query_words if w in text_lower)
            # Jaccard index style normalization
            score = match_count / (len(query_words) + 1e-9)
            scores.append(score)
            
        top_indices = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                item = self.chunks[int(idx)].copy()
                item["score"] = float(scores[idx])
                results.append(item)
        return results

# Initialize global RAG instance
rag_assistant = RAGEngine()
