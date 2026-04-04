from pathlib import Path
from document import ParsedDocument,DocumentType

class DocumentParser:

    def parse(self,source: str| Path) -> ParsedDocument:

        pass


    def _parse_pdf(self, path: Path):
        pass

    def _parse_html(self, path: Path) -> tuple[str,list,dict]:
        pass

    def _parse_txt(self,path:Path) -> tuple[str,list,dict]:
        pass

    def _infer_metadata(self,doc:ParsedDocument,path:Path):
        pass


