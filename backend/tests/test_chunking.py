from backend.ingestion.chunking.fixed import FixedChunker
from backend.ingestion.chunking.sentence import SentenceChunker
from backend.ingestion.chunking.metadata_aware import MetadataAwareChunker

def test_fixed_chunker():
    chunker = FixedChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("01234567890123456789", document_id="doc1")
    assert len(chunks) == 3
    assert chunks[0].text == "0123456789"
    assert chunks[0].chunk_strategy == "fixed"

def test_sentence_chunker():
    chunker = SentenceChunker(max_chunk_size=50, overlap_sentences=1)
    chunks = chunker.chunk("This is sentence one. This is sentence two. And this is the third sentence.", document_id="doc1")
    assert len(chunks) >= 2
    assert chunks[0].chunk_strategy == "sentence"

def test_metadata_aware_chunker():
    chunker = MetadataAwareChunker(max_chunk_size=15)
    chunks = chunker.chunk("Paragraph one.\n\nParagraph two.", document_id="doc1", metadata={"title": "Test"})
    assert len(chunks) >= 2
    assert chunks[0].metadata["title"] == "Test"
    assert chunks[0].chunk_strategy == "metadata_aware"
