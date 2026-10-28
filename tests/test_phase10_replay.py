from pathlib import Path
import json
from scripts.phase10_replay_tokens import replay


def test_replay_realistic_phase09_log(tmp_path):
    tokens=tmp_path/'tokens.jsonl'
    tokens.write_text('\n'.join([
        json.dumps({'event':'TOKEN','label':'xin_chao'}),
        json.dumps({'event':'TOKEN','label':'cam_on'}),
        json.dumps({'event':'TOKEN','label':'REST'}),
        json.dumps({'event':'TOKEN','label':'UNKNOWN'}),
        json.dumps({'event':'TRIAL_START','label':'xin_loi'}),
        json.dumps({'event':'TOKEN','label':'xin_loi'})
    ]),encoding='utf-8')
    result=replay(tokens,{'phrases':{'xin_chao':'xin chào','cam_on':'cảm ơn','xin_loi':'xin lỗi'}},tmp_path/'sessions')
    assert result['accepted']==3
    assert result['ignored']==2
    assert result['sentences']==1
    assert Path(result['transcript']).read_text(encoding='utf-8').strip()=='Xin chào cảm ơn xin lỗi.'


def test_replay_periodic_finalize(tmp_path):
    p=tmp_path/'x.jsonl'
    p.write_text('\n'.join(json.dumps({'event':'TOKEN','label':x}) for x in ['xin_chao','cam_on','xin_loi']),encoding='utf-8')
    r=replay(p,{},tmp_path/'sessions',finalize_every=2)
    assert r['sentences']==2
    assert r['preview']=='Xin chào cảm ơn.\nXin lỗi.'
