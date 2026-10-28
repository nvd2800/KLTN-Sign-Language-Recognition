"""Guard the exact Phase09 event interception contract of this patch."""
from pathlib import Path


def test_live_phase09_only_adds_confirmed_tokens():
    source=Path(__file__).resolve().parents[1]/'scripts'/'live_phase09_v3_spatial.py'
    text=source.read_text(encoding='utf-8')
    assert 'event=engine.update(label,conf,ts)' in text
    assert 'if event is not None:' in text
    assert 'if phase10_builder.add(event.label):' in text
    assert '--phase10-config' in text
    assert 'finalize_phase10(speak_if_enabled=False)' in text


def test_launcher_passes_phase10_config():
    source=Path(__file__).resolve().parents[1]/'scripts'/'live_phase10_v3_1.py'
    text=source.read_text(encoding='utf-8')
    assert "forwarded += ['--phase10-config',str(cfg)]" in text
    assert 'from scripts.live_phase09_v3_1 import launch' in text
