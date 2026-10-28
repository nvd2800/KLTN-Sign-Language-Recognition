import io
import json
import threading
import wave
from pathlib import Path
from urllib.request import Request, urlopen
import pytest

from core.phase10_tts import VietnameseSpeaker, prepare_speech_text
from scripts.phase10_vieneu_server import create_server


class FakeEngine:
    def __init__(self):
        self.calls=[]

    def infer(self, text, *, voice):
        self.calls.append((text,voice))
        return b'123'

    def save(self, audio, path):
        assert audio == b'123'
        with wave.open(path, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(b'\x00\x00'*2400)


@pytest.fixture
def service():
    e=FakeEngine()
    server=create_server(e,port=0,voice_name='Minh Đức')
    thread=threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:yield (server.server_address[1],e)
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_vieneu_local_utf8_full_sentence(service,tmp_path):
    port,engine=service
    obj=VietnameseSpeaker('vieneu',vieneu_endpoint=f'http://127.0.0.1:{port}/synthesize',
                         vieneu_voice='Minh Đức', wav_output_dir=tmp_path,player=lambda _:True)
    assert obj.info.available
    sentence='Xin chào. Cảm ơn bạn. Xin lỗi!'
    assert obj.speak(sentence)==(True,'SPOKEN')
    assert engine.calls==[(sentence,'Minh Đức')]
    assert obj.last_wav.exists()
    with wave.open(str(obj.last_wav),'rb') as wf:
        assert wf.getframerate()==24000
        assert wf.getnframes()==2400


def test_vieneu_responds_health(service):
    port,_=service
    with urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as r:
        result=json.loads(r.read().decode('utf-8'))
        assert result['ready'] is True
        assert result['local_only'] is True


def test_vieneu_blocks_wrong_voice(service,tmp_path):
    port,_=service
    speaker=VietnameseSpeaker('vieneu',vieneu_endpoint=f'http://127.0.0.1:{port}/synthesize',
                              vieneu_voice='Wrong Voice',wav_output_dir=tmp_path,play_audio=False)
    ok,status=speaker.speak('Xin chào.')
    assert not ok
    assert 'VOICE_MISMATCH' in status


def test_vieneu_rejects_untrusted_endpoint():
    with pytest.raises(ValueError,match='VIENEU_ENDPOINT_NOT_LOCAL'):
        VietnameseSpeaker('vieneu',vieneu_endpoint='https://example.com/synthesize')


def test_vieneu_unavailable_does_not_fall_back(tmp_path):
    speaker=VietnameseSpeaker('vieneu',vieneu_endpoint='http://127.0.0.1:59337/synthesize',
                              wav_output_dir=tmp_path)
    assert not speaker.info.available
    assert speaker.speak('Xin chào.')[0] is False


def test_sentences_stay_in_full_words():
    assert prepare_speech_text('xin_chao cam_on xin_loi')=='Xin chào cảm ơn xin lỗi.'
    assert prepare_speech_text('Xin chào bạn!')=='Xin chào bạn!'
