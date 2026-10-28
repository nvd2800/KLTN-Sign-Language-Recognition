import json
from pathlib import Path
import pytest

from core.phase10_sentence import SentenceBuilder
from core.phase10_store import Phase10Store
from core.phase10_tts import VietnameseSpeaker,SpeechQueue


def test_build_three_phrases():
    b=SentenceBuilder()
    assert b.add('xin_chao') and b.add('cam_on') and b.add('xin_loi')
    assert b.preview=='Xin chào cảm ơn xin lỗi'
    s=b.finalize(finalized_utc='2026-10-28T00:00:00Z')
    assert s.text=='Xin chào cảm ơn xin lỗi.'
    assert s.tokens==('xin_chao','cam_on','xin_loi')
    assert b.preview==''


def test_no_unknown_rest_or_invalid():
    b=SentenceBuilder()
    assert not b.add('REST')
    assert not b.add('UNKNOWN')
    assert not b.add('other_word')
    assert b.finalize() is None


def test_repeat_preservation():
    b=SentenceBuilder()
    b.add('xin_chao');b.add('xin_chao')
    assert b.finalize().text=='Xin chào xin chào.'


def test_undo_clear_history():
    b=SentenceBuilder()
    b.add('xin_loi');b.finalize()
    b.add('cam_on');assert b.undo()=='cam_on'
    assert b.undo() is None
    b.add('xin_chao');b.clear_active()
    assert len(b.sentences)==1 and b.last_text()=='Xin lỗi.'


def test_max_tokens_and_validation():
    with pytest.raises(ValueError):SentenceBuilder({'REST':'nghỉ'})
    with pytest.raises(ValueError):SentenceBuilder(max_tokens=0)
    b=SentenceBuilder(max_tokens=1)
    assert b.add('cam_on') and not b.add('cam_on')


def test_session_files_vietnamese(tmp_path):
    s=Phase10Store(tmp_path,session_id='SES01')
    b=SentenceBuilder();b.add('cam_on');sentence=b.finalize()
    s.sentence(sentence);s.append('TTS_REQUEST',queued=False)
    s.export(b)
    assert 'Cảm ơn.' in s.transcript_path.read_text(encoding='utf-8')
    snap=json.loads(s.snapshot_path.read_text(encoding='utf-8'))
    assert snap['sentences'][0]['tokens']==['cam_on']
    assert len(s.event_path.read_text(encoding='utf-8').splitlines())==2
    with pytest.raises(FileExistsError):Phase10Store(tmp_path,session_id='SES01')


def test_invalid_session_id(tmp_path):
    with pytest.raises(ValueError):Phase10Store(tmp_path,session_id='../escape')


def test_tts_none_is_graceful():
    sp=VietnameseSpeaker('none')
    assert not sp.info.available
    ok,status=sp.speak('Xin chào')
    assert (ok,status)==(False,'VOICE_UNAVAILABLE')
    q=SpeechQueue(sp)
    assert not q.say('Xin chào')
    q.close()


def test_tts_windows_voice_not_present():
    sp=VietnameseSpeaker('windows',platform='linux',which=lambda p:None)
    assert not sp.info.available


def test_tts_no_untrusted_shell():
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        class Result:
            stdout='vi-VN|Microsoft Vietnam Voice\n'
            stderr=''
            returncode=0
        return Result()
    sp=VietnameseSpeaker('windows',runner=run,platform='win32',which=lambda p:True)
    assert sp.info.available
    ok,status=sp.speak('Xin chào; rm -rf /')
    assert ok and status=='SPOKEN'
    assert calls[-1][0]=='powershell' and len(calls[-1])==5
    assert 'rm -rf' not in calls[-1][-1] # only base64 encoded text embedded


def test_tts_espeak_detection():
    def run(args,**kw):
        class Result:
            stdout='Pty Language Age/Gender VoiceName\n 5 vi --/M Vietnamese\n'
            returncode=0
            stderr=''
        return Result()
    sp=VietnameseSpeaker('espeak',runner=run,which=lambda exe:True)
    assert sp.info.available
    assert sp.speak('Xin chào')[0]


def test_punctuation_multiple_sentences():
    b=SentenceBuilder({'abc':'đây là thử nghiệm'})
    b.add('abc'); b.finalize();b.add('abc');b.finalize()
    assert b.all_text()=='Đây là thử nghiệm.\nĐây là thử nghiệm.'


def test_builder_snapshot():
    b=SentenceBuilder()
    b.add('xin_chao')
    assert b.snapshot()['preview']=='Xin chào'
    assert b.snapshot()['active_tokens']==['xin_chao']
