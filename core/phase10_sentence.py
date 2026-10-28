"""Phase10.1: deterministic Vietnamese phrase assembly from Phase09 *confirmed* tokens.

This is a transparent phrase assembler, NOT semantic machine translation.
New label mappings are pending until the TCN classifier actually outputs that label.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping
import re

DEFAULT_PHRASES = {'xin_chao':'xin chào','cam_on':'cảm ơn','xin_loi':'xin lỗi'}
EXCLUDED_LABELS = {'REST','UNKNOWN'}
PUNCTUATION_ALIASES = {
    'dau_phay': ',', 'dau_cham': '.', 'dau_hoi': '?', 'dau_cam': '!',
    'dau_cham_phay': ';', 'dau_hai_cham': ':'
}
PUNCTUATION = frozenset(',.:;!?')
TERMINAL = frozenset('.!?')
_MARKER = '<punct:'


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str
    tokens: tuple[str, ...]
    finalized_utc: str


class SentenceBuilder:
    """Compatible with Phase10 V1 API (`add/undo/clear_active/finalize/preview`).

    `active` stores labels and internal punctuation markers.  No language model
    invents words.  Repeated tokens remain repeated; punctuation is optional.
    """
    def __init__(self, phrases: Mapping[str,str] | None=None, *, max_tokens: int=40,
                 punctuation_labels: Mapping[str,str] | None=None):
        self.phrases = dict(DEFAULT_PHRASES if phrases is None else phrases)
        if not self.phrases or any(not isinstance(k,str) or not k.strip() or
                                   not re.fullmatch(r'[\w\-]+', k) or
                                   not isinstance(v,str) or not v.strip()
                                   for k,v in self.phrases.items()):
            raise ValueError('Nonempty mapping label -> Vietnamese phrase required')
        if EXCLUDED_LABELS & self.phrases.keys():
            raise ValueError('REST and UNKNOWN must not map to text')
        if not 1 <= int(max_tokens) <= 500:
            raise ValueError('max_tokens must be between 1 and 500')
        self.max_tokens = int(max_tokens)
        self.punctuation_labels = dict(PUNCTUATION_ALIASES if punctuation_labels is None else punctuation_labels)
        if any(k in EXCLUDED_LABELS or k in self.phrases or v not in PUNCTUATION
               for k,v in self.punctuation_labels.items()):
            raise ValueError('Punctuation label collision or invalid mark')
        self.active: list[str] = []
        self.sentences: list[Sentence] = []

    @staticmethod
    def _is_punct(token:str) -> bool:
        return token.startswith(_MARKER) and token.endswith('>')

    @staticmethod
    def _punct(mark:str) -> str:
        return _MARKER + mark + '>'

    def add(self,label:str) -> bool:
        if label in EXCLUDED_LABELS or len(self.active)>=self.max_tokens:
            return False
        if label in self.punctuation_labels:
            return self.add_punctuation(self.punctuation_labels[label])
        if label not in self.phrases:
            return False
        self.active.append(label)
        return True

    def add_punctuation(self,mark:str) -> bool:
        if mark not in PUNCTUATION or not self.active or len(self.active)>=self.max_tokens:
            return False
        if self._is_punct(self.active[-1]):
            # Replace the last punctuation instead of generating '?!' or ',.'
            self.active[-1] = self._punct(mark)
        else:
            self.active.append(self._punct(mark))
        return True

    def undo(self) -> str | None:
        return self.active.pop() if self.active else None

    def clear_active(self) -> None:
        self.active.clear()

    @staticmethod
    def _cap_first(text:str) -> str:
        return text[0].upper()+text[1:] if text else text

    @classmethod
    def _render(cls,active:list[str],phrases:Mapping[str,str],*,end_mark:bool) -> str:
        buf=''; capitalize_next=True
        for item in active:
            if cls._is_punct(item):
                p=item[len(_MARKER):-1]
                buf=buf.rstrip() + p
                capitalize_next=p in TERMINAL
            else:
                phrase=phrases[item].strip()
                if capitalize_next:
                    phrase=cls._cap_first(phrase)
                    capitalize_next=False
                buf+=(' ' if buf else '')+phrase
        if end_mark and buf and buf[-1] not in TERMINAL:
            buf+='.'
        return buf

    @property
    def preview(self) -> str:
        return self._render(self.active,self.phrases,end_mark=False)

    def finalize(self, *, finalized_utc: str | None=None, terminal_mark:str='.') -> Sentence | None:
        if not self.active:
            return None
        if terminal_mark not in TERMINAL:
            raise ValueError('terminal_mark must be . ! or ?')
        text=self._render(self.active,self.phrases,end_mark=False)
        if text[-1] not in TERMINAL:
            text+=terminal_mark
        labels=tuple(i for i in self.active if not self._is_punct(i))
        if not labels:
            return None
        event=Sentence(len(self.sentences)+1,text,labels,finalized_utc or datetime.now(timezone.utc).isoformat())
        self.sentences.append(event)
        self.active.clear()
        return event

    def last_text(self) -> str:
        return self.sentences[-1].text if self.sentences else ''

    def all_text(self, *, include_active:bool=False) -> str:
        lines=[s.text for s in self.sentences]
        if include_active and self.preview:
            lines.append(self.preview)
        return '\n'.join(lines)

    def snapshot(self) -> dict:
        return {'active_tokens':[t for t in self.active if not self._is_punct(t)],
                'active_items':list(self.active),
                'preview':self.preview,
                'sentences':[{'index':s.index,'text':s.text,'tokens':list(s.tokens),
                              'finalized_utc':s.finalized_utc} for s in self.sentences]}
