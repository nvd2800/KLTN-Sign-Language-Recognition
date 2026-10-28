import json
from pathlib import Path
from unittest.mock import Mock
import yaml
import pytest
from core.phase10_sentence import SentenceBuilder
from core.phase10_tts import VietnameseSpeaker,SpeechQueue
from scripts.phase10_manage_vocabulary import apply,read_labels


def test_punctuation_spacing_capitalization():
    b=SentenceBuilder({'xin_chao':'xin chào','cam_on':'cảm ơn','xin_loi':'xin lỗi'})
    assert b.add('xin_chao')
    assert b.add_punctuation(',')
    assert b.add('cam_on')
    assert b.add_punctuation('!')
    assert b.add('xin_loi')
    assert b.finalize().text=='Xin chào, cảm ơn! Xin lỗi.'


def test_question_and_exclaim_final():
    b=SentenceBuilder()
    b.add('xin_chao');b.add_punctuation('?')
    assert b.finalize().text=='Xin chào?'
    b.add('xin_loi');assert b.finalize(terminal_mark='!').text=='Xin lỗi!'


def test_punct_tokens_not_plain_phrases():
    b=SentenceBuilder()
    assert not b.add('dau_hoi')
    b.add('xin_chao')
    assert b.add('dau_phay')
    assert b.preview=='Xin chào,'
    assert b.add('dau_hoi')
    assert b.preview=='Xin chào?'
    assert b.undo()=='<punct:?>'
    assert b.preview=='Xin chào'
    assert not b.add('REST') and not b.add('UNKNOWN')


def test_three_repeat_preserved():
    b=SentenceBuilder()
    for _ in range(3):assert b.add('xin_chao')
    assert b.finalize().text=='Xin chào xin chào xin chào.'


def test_punctuation_invalid_and_limit():
    b=SentenceBuilder(max_tokens=2)
    assert not b.add_punctuation(',')
    b.add('xin_chao');b.add_punctuation(',')
    assert not b.add('cam_on')
    assert not b.add_punctuation('^')
    with pytest.raises(ValueError):b.finalize(terminal_mark=';')


def test_pending_not_auto_active():
    cfg={'phrases':{'xin_chao':'xin chào'},'vocabulary_pending':{'toi':'tôi'}}
    b=SentenceBuilder(cfg['phrases'])
    assert not b.add('toi')
    with pytest.raises(ValueError,match='UNTRAINED_LABEL'):
        apply(cfg,'activate','toi',model_labels={'xin_chao'})
    apply(cfg,'activate','toi',model_labels={'xin_chao','toi'})
    assert b.add('xin_chao')
    assert SentenceBuilder(cfg['phrases']).add('toi')


def test_add_pending_and_deactivate():
    cfg={'phrases':{'cam_on':'cảm ơn'},'vocabulary_pending':{}}
    apply(cfg,'add','ho_tro','hỗ trợ')
    assert cfg['vocabulary_pending']['ho_tro']=='hỗ trợ'
    apply(cfg,'activate','ho_tro',model_labels={'ho_tro'})
    apply(cfg,'deactivate','ho_tro')
    assert 'ho_tro' not in cfg['phrases']
    with pytest.raises(ValueError):apply(cfg,'add','INVALID LABEL','abc')


def test_offline_piper_detection_requires_local_model(tmp_path):
    voice=tmp_path/'voice.onnx';voice.write_bytes(b'mock')
    sp=VietnameseSpeaker('piper',model_path=voice,which=lambda name:'/bin/piper')
    assert not sp.info.available
    Path(str(voice)+'.json').write_text('{}')
    sp=VietnameseSpeaker('piper',model_path=voice,which=lambda name:'/bin/piper')
    assert sp.info.available


def test_piper_subprocess_utf8_no_shell_and_wav(tmp_path):
    voice=tmp_path/'voice.onnx';voice.write_bytes(b'mock')
    Path(str(voice)+'.json').write_text('{}')
    calls=[]
    def runner(cmd,**kw):
        calls.append((cmd,kw))
        wav=Path(cmd[cmd.index('--output_file')+1])
        wav.write_bytes(b'RIFF'+b'\x00'*48)
        class R:returncode=0
        return R()
    player=lambda p:p.is_file()
    sp=VietnameseSpeaker('piper',model_path=voice,wav_output_dir=tmp_path/'out',
                         runner=runner,which=lambda name:'/bin/piper',player=player)
    ok,status=sp.speak('Xin chào; $(bad)')
    assert ok and status=='SPOKEN'
    assert calls[0][0][:2]==['piper','--model']
    assert calls[0][1]['input']==('Xin chào; $(bad)\n').encode('utf-8')
    assert sp.last_wav.is_file()


def test_piper_bad_wav_does_not_claim_speak(tmp_path):
    voice=tmp_path/'voice.onnx';voice.write_bytes(b'mock')
    Path(str(voice)+'.json').write_text('{}')
    def runner(cmd,**kw):
        Path(cmd[-1]).write_bytes(b'BAD')
        class R:returncode=0
        return R()
    sp=VietnameseSpeaker('piper',model_path=voice,wav_output_dir=tmp_path/'out',
                         runner=runner,which=lambda name:'piper')
    assert sp.speak('xin chào')==(False,'PIPER_INVALID_WAV')


def test_auto_prefers_neural_then_fallback(tmp_path):
    voice=tmp_path/'voice.onnx';voice.write_bytes(b'mock')
    Path(str(voice)+'.json').write_text('{}')
    sp=VietnameseSpeaker('auto',model_path=voice,which=lambda n:'piper')
    assert sp.info.backend=='piper'


def test_untrained_punct_labels_do_not_emit_by_default():
    b=SentenceBuilder({'xin_chao':'xin chào'},punctuation_labels={})
    assert not b.add('dau_hoi')


def test_snapshot_keeps_labels_only():
    b=SentenceBuilder();b.add('xin_chao');b.add_punctuation('!')
    snap=b.snapshot()
    assert snap['active_tokens']==['xin_chao']
    assert snap['active_items']==['xin_chao','<punct:!>']
