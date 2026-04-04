import re
from document import ParsedDcoument

_AMOUNT_BILLION = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)\s*billion', re.IGNORECASE)
_AMOUNT_MILLION = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)\s*million', re.IGNORECASE)
_PERCENT_WORD   = re.compile(r'([\d.]+)\s*percent', re.IGNORECASE)
_HTML_TAG       = re.compile(r'<[^>]+>')
_UNICODE_DASH   = re.compile(r'[\u2013\u2014\u2212]') 


class DocumentCleaner:

    BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r'this (report|filing) contains forward.looking statements.{0,300}risks', re.IGNORECASE | re.DOTALL),
    re.compile(r'safe\s+harbor\s+statement.{0,500}', re.IGNORECASE | re.DOTALL),
    re.compile(r'(all rights reserved|©\s*\d{4}).{0,100}', re.IGNORECASE),
]
        

    def clean(self, doc: ParsedDocument) -> ParsedDcoument:
        pass

    def _strip_html(self, text:str) -> str:
        pass

    def _remove_boilerplate(self,text: str)-> str:
        pass

    def _normalise_amounts(self, text: str) -> str:
        pass
    def _fix_unicode(self,text:str) -> str:
        pass

    

