from pathlib import Path
from core.phase10_sentence import SentenceBuilder
from core.phase10_tts import VietnameseSpeaker, SpeechQueue, prepare_speech_text
import pytest


def test_full_sentence_with_diacritics_and_punctuation():
    b=SentenceBuilder()
    assert all(b.add(t) for t in ('xin_chao','cam_on','xin_loi'))
    assert b.finalize().text=='Xin chào cảm ơn xin lỗi.'
    assert prepare_speech_text(b.last_text())=='Xin chào cảm ơn xin lỗi.'


def test_old_unrendered_labels_become_full_phrases():
    assert prepare_speech_text('xin_chao cam_on xin_loi')=='Xin chào cảm ơn xin lỗi.'


def test_spaced_characters_and_unknown_label_rejected():
    with pytest.raises(ValueError,match='CHARACTER_SPLIT'):
        prepare_speech_text('x i n c h a o')
    with pytest.raises(ValueError,match='UNRENDERED_LABEL'):
        prepare_speech_text('some_other_sign')


def test_piper_receives_one_whole_utf8_sentence(tmp_path):
    model=tmp_path/'vi.onnx';model.write_bytes(b'fake')
    Path(str(model)+'.json').write_text('{}')
    calls=[]
    class Result:returncode=0
    def run(cmd,**kwargs):
        calls.append((cmd,kwargs))
        Path(cmd[-1]).write_bytes(b'RIFF'+b'\0'*50)
        return Result()
    speaker=VietnameseSpeaker('piper',model_path=model,wav_output_dir=tmp_path/'audio',runner=run,
                              which=lambda cmd:'/mock/piper',player=lambda wav:True)
    txt='Xin chào. Cảm ơn bạn! Xin lỗi.'
    assert speaker.speak(txt)==(True,'SPOKEN')
    assert len(calls)==1
    assert calls[0][1]['input']==(txt+'\n').encode('utf-8')
    assert calls[0][0][0]=='piper'


def test_missing_piper_does_not_silently_use_espeak_or_windows():
    # In auto, even if other synth commands are present, a voice must be selected explicitly.
    speaker=VietnameseSpeaker('auto',model_path='/path/does-not-exist.onnx',
                              which=lambda cmd: '/mock/'+cmd,platform='win32')
    assert not speaker.info.available
    assert speaker.info.backend=='auto'
    assert speaker.speak('Xin chào.')[0] is False


def test_queue_rejects_character_split_before_thread_starts(tmp_path):
    model=tmp_path/'model.onnx';model.write_bytes(b'fake')
    Path(str(model)+'.json').write_text('{}')
    speaker=VietnameseSpeaker('piper',model_path=model,which=lambda c:'/mock/piper')
    queue=SpeechQueue(speaker)
    assert queue.say('x i n c h a o') is False
    assert 'CHARACTER_SPLIT' in queue.last_status
    assert queue._thread is None


def test_nfc_normalization():
    assert prepare_speech_text('Ca\u0309m ơn')=='Cảm ơn'
