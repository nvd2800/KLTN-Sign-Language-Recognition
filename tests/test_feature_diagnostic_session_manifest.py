from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from core.feature_diagnostic_video import (
    save_source_video, probe_video, write_metadata, append_trial_manifest,
    rebuild_trial_manifest, MANIFEST_COLUMNS,
)
from scripts.live_feature_diagnostic import parse_args
from scripts.review_feature_diagnostic_videos import collect_rows,render_contact_sheet,record_manual_decision
from scripts.collect_v3_train_negative import validate_capture
from scripts.rebuild_feature_diagnostic_manifest import main as rebuild_main


def sample(tmp_path, condition='wrong_position', session='SES02', with_video=True):
    root=tmp_path/'logs'/'feature_diagnostic'
    trial=root/'xin_chao'/'S01'/condition/'20261028T081234_654321Z'
    trial.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trial/'raw_landmarks.npz',dummy=np.ones((12,21,3),dtype=np.float32))
    np.savez_compressed(trial/'features.npz',features453=np.ones((12,453),dtype=np.float32))
    meta={'sample_id':trial.name,'captured_at':'2026-10-28T08:12:34+00:00',
          'label':'xin_chao','subject':'S01','subject_id':'S01',
          'session':session,'session_id':session,'condition':condition,
          'frames':12,'qc':{'any_hand_ratio':.8,'pose_ratio':.95}}
    if with_video:
        frames=[]
        for idx in range(12):
            frame=np.full((80,128,3),(idx*13%255,50,80),np.uint8)
            cv2.circle(frame,(12+5*idx,35),8,(255,255,255),-1)
            frames.append(frame)
        meta['video']=save_source_video(trial,frames,[i/25 for i in range(12)])
    write_metadata(trial,meta)
    return root,trial,meta


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def test_cli_session_and_no_alias():
    a=parse_args(['--label','xin_chao','--subject','S01','--session','SES03','--frames','32'])
    assert a.session=='SES03' and a.save_video is True
    assert not hasattr(a,'signer_alias')
    assert parse_args([]).session=='SES01'
    assert parse_args(['--no-video']).save_video is False
    assert parse_args(['--video-format','mp4']).video_format=='mp4'
    with pytest.raises(SystemExit):parse_args(['--signer-alias','alias'])


def test_compatible_layout_session_and_video(tmp_path):
    root,trial,meta=sample(tmp_path)
    assert trial.relative_to(root).parts==('xin_chao','S01','wrong_position','20261028T081234_654321Z')
    assert {p.name for p in trial.iterdir()} == {'raw_landmarks.npz','features.npz','metadata.json','source_video.avi'}
    assert meta['session']=='SES02' and meta['subject']=='S01'
    assert 'signer_alias' not in meta
    assert probe_video(trial/'source_video.avi',expected_frames=12)['decoded_frames']==12


def test_manifest_new_row_with_video(tmp_path):
    root,trial,meta=sample(tmp_path)
    path=append_trial_manifest(root,trial,meta)
    assert path==root/'manifest.csv'
    recorded=rows(path)
    assert len(recorded)==1
    assert list(recorded[0])==MANIFEST_COLUMNS
    assert 'signer_alias' not in recorded[0]
    assert recorded[0]['session']=='SES02' and recorded[0]['subject']=='S01'
    assert recorded[0]['sample_id']==trial.name
    assert recorded[0]['video_status']=='SAVED_VERIFIED'
    assert recorded[0]['video_frames']=='12'
    assert Path(recorded[0]['video_file']).is_file()
    assert recorded[0]['video_sha256']==meta['video']['sha256']


def test_manifest_upsert_no_duplicate(tmp_path):
    root,trial,meta=sample(tmp_path)
    append_trial_manifest(root,trial,meta)
    append_trial_manifest(root,trial,meta)
    assert len(rows(root/'manifest.csv'))==1


def test_manifest_two_sessions_same_signer(tmp_path):
    root,trial,meta=sample(tmp_path)
    append_trial_manifest(root,trial,meta)
    other=root/'xin_chao'/'S01'/'wrong_handshape'/'20261028T091234_654321Z'
    other.mkdir(parents=True)
    meta2={**meta,'sample_id':other.name,'session':'SES03','session_id':'SES03',
           'condition':'wrong_handshape','video':{'status':'DISABLED'}}
    write_metadata(other,meta2)
    append_trial_manifest(root,other,meta2)
    recorded=rows(root/'manifest.csv')
    assert len(recorded)==2
    assert [r['session'] for r in recorded]==['SES02','SES03']
    assert all(r['subject']=='S01' for r in recorded)


def test_existing_v2_alias_manifest_auto_migrates_without_losing_rows(tmp_path):
    root,trial,meta=sample(tmp_path)
    path=root/'manifest.csv'
    oldcols=MANIFEST_COLUMNS.copy();oldcols.insert(oldcols.index('session'),'signer_alias')
    record={k:'' for k in oldcols}
    record.update(sample_id='OLD_SAMPLE',subject='S02',signer_alias='test',session='SES00',
                  label='cam_on',condition='partial',trial_dir=str(root/'cam_on'/'S02'/'partial'/'OLD_SAMPLE'))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=oldcols)
        writer.writeheader();writer.writerow(record)
    oldbytes=path.read_bytes()
    append_trial_manifest(root,trial,meta)
    assert (root/'manifest.v2_alias_backup.csv').read_bytes()==oldbytes
    newrows=rows(path)
    assert len(newrows)==2
    assert {r['sample_id'] for r in newrows}=={'OLD_SAMPLE',trial.name}
    assert all('signer_alias' not in r for r in newrows)
    assert newrows[0]['session']=='SES00' and newrows[1]['session']=='SES02'


def test_unknown_manifest_schema_is_not_overwritten(tmp_path):
    root,trial,meta=sample(tmp_path)
    p=root/'manifest.csv';p.write_text('invalid,schema\na,b\n',encoding='utf-8')
    before=p.read_bytes()
    with pytest.raises(ValueError,match='Unknown manifest columns'):
        append_trial_manifest(root,trial,meta)
    assert p.read_bytes()==before


def test_rebuild_legacy_and_new_preserves_raw(tmp_path):
    root,trial,meta=sample(tmp_path)
    legacy=root/'cam_on'/'S02'/'wrong_position'/'20260901T020101Z'
    legacy.mkdir(parents=True)
    np.savez_compressed(legacy/'raw_landmarks.npz',dummy=np.zeros((8,21,3),np.float32))
    np.savez_compressed(legacy/'features.npz',dummy=np.zeros((8,453),np.float32))
    legacy_meta={'sample_id':legacy.name,'label':'cam_on','subject':'S02',
                 'condition':'wrong_position','frames':8}
    write_metadata(legacy,legacy_meta)
    oldraw=(legacy/'raw_landmarks.npz').read_bytes()
    outcome=rebuild_trial_manifest(root)
    assert outcome['scanned']==2 and outcome['total']==2 and not outcome['skipped']
    entries=rows(root/'manifest.csv')
    assert len(entries)==2
    absent=[r for r in entries if r['sample_id']==legacy.name][0]
    assert absent['video_status']=='LEGACY_VIDEO_MISSING'
    assert absent['session']==''
    assert (legacy/'raw_landmarks.npz').read_bytes()==oldraw
    assert rebuild_trial_manifest(root)['total']==2


def test_rebuild_cli(tmp_path):
    root,trial,meta=sample(tmp_path)
    assert rebuild_main(['--input',str(root)])['total']==1


def test_rebuild_skips_wrong_subject_directory(tmp_path):
    root,trial,meta=sample(tmp_path)
    meta['subject']='S09'
    write_metadata(trial,meta)
    result=rebuild_trial_manifest(root)
    assert result['total']==0 and len(result['skipped'])==1


def test_reviewer_session_and_contact_sheet(tmp_path):
    root,trial,meta=sample(tmp_path)
    entries=collect_rows(root,label='xin_chao',subject='S01')
    assert len(entries)==1
    assert entries[0]['status']=='PASS'
    assert entries[0]['session']=='SES02'
    assert 'signer_alias' not in entries[0]
    target=tmp_path/'review.jpg'
    assert render_contact_sheet(entries[0],target)
    assert target.is_file()
    record_manual_decision(tmp_path/'review.csv',entries[0],'ACCEPT','GV01')
    assert rows(tmp_path/'review.csv')[0]['session']=='SES02'
    assert json.loads((trial/'metadata.json').read_text(encoding='utf-8'))==meta


def test_old_no_video_review(tmp_path):
    root,trial,meta=sample(tmp_path,with_video=False)
    assert 'LEGACY_VIDEO_MISSING' in collect_rows(root)[0]['problem']


def test_train_signer_guard():
    mapping={'S01':'train','S08':'val','S09':'test'}
    labels=['REST','cam_on','xin_chao','xin_loi','UNKNOWN']
    assert validate_capture('xin_chao','S01',mapping,labels)==('xin_chao','S01')
    with pytest.raises(ValueError):validate_capture('xin_chao','S08',mapping,labels)
    with pytest.raises(ValueError):validate_capture('xin_chao','S09',mapping,labels)


def test_manifest_keeps_video_failure(tmp_path):
    root,trial,meta=sample(tmp_path,with_video=False)
    meta['video']={'status':'FAILED','filename':None,'error':'codec failed'}
    write_metadata(trial,meta)
    append_trial_manifest(root,trial,meta)
    assert rows(root/'manifest.csv')[0]['video_status']=='FAILED'


def test_manifest_does_not_alter_legacy_metadata(tmp_path):
    root,trial,meta=sample(tmp_path)
    before=(trial/'metadata.json').read_bytes()
    append_trial_manifest(root,trial,meta)
    assert (trial/'metadata.json').read_bytes()==before
