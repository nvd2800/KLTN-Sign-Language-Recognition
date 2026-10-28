"""Durable, isolated Phase10 session logs. Never modify Phase09 logs or checkpoints."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import uuid


class Phase10Store:
    def __init__(self, root: str | Path, *, session_id: str | None = None):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        if session_id is None:
            session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        if not re.fullmatch(r'[A-Za-z0-9_\-]{1,80}',session_id):
            raise ValueError('Unsafe session id')
        self.session_id=session_id
        self.session_dir=self.root/session_id
        # Refuse to append unrelated data to an existing session.
        self.session_dir.mkdir(parents=True,exist_ok=False)
        self.event_path=self.session_dir/'events.jsonl'
        self.sentences_path=self.session_dir/'sentences.jsonl'
        self.transcript_path=self.session_dir/'transcript.txt'
        self.snapshot_path=self.session_dir/'session_snapshot.json'

    def append(self, event: str, **fields) -> None:
        record={'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'event':event,'session_id':self.session_id,**fields}
        with self.event_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')

    def sentence(self, sentence) -> None:
        record={'index':sentence.index,'text':sentence.text,'tokens':list(sentence.tokens),'finalized_utc':sentence.finalized_utc,'session_id':self.session_id}
        with self.sentences_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.append('SENTENCE_FINALIZED',**record)

    def export(self,builder) -> None:
        self.transcript_path.write_text(builder.all_text()+'\n' if builder.sentences else '',encoding='utf-8')
        snapshot=builder.snapshot();snapshot['session_id']=self.session_id
        self.snapshot_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
