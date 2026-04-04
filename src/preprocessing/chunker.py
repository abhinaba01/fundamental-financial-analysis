from document import ParsedDocument,DocumentChunk,ChunkType

MAX_TOKENS = 512
OVERLAP_TOKENS = 64

class SemanticChunker:

    def chunk(self, doc: ParsedDocument) -> ParsedDocument:
        pass

    def _chunk_section(self, text:str,chunk_type:ChunkType) ->list[DocumentChunk]:
        pass

    def _count_tokens(self,text: str) -> int:
        pass

