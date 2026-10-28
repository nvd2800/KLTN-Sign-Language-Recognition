from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from core.feature_diagnostic_video import (
    next_sample_path, write_metadata, append_trial_manifest,
    check_trial_layout, rebuild_trial_manifest,
)
from scripts.migrate_feature_diagnostic_layout import plan, migrate
from scripts.review_feature_diagnostic_videos import collect_rows


def create(root, label='cam_on', subject='S07', session='SES01', condition='wrong_handshape', video=False):
    dest, sid = next_sample_path(root, label, subject, session)
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'raw_landmarks.npz', dummy=np.ones((32, 21, 3)))
    np.savez_compressed(dest/'features.npz', features453=np.ones((32,453)))
    meta={'sample_id': sid, 'label': label, 'subject': subject, 'subject_id':subject,
          'session': session, 'session_id':session, 'condition':condition,
          'frames':32,'qc':{'any_hand_ratio':.9},'video':{'status':'DISABLED'}}
    write_metadata(dest,meta)
    append_trial_manifest(root,dest,meta)
    return dest,meta


def manifest_rows(root):
    with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def test_exact_screenshot_layout_and_increment(tmp_path):
    root=tmp_path/'feature_diagnostic'
    a,ma=create(root,condition='correct')
    b,mb=create(root,condition='wrong_position')
    assert a.relative_to(root).parts==('cam_on','S07','SES01','cam_on_S07_SES01_001')
    assert b.relative_to(root).parts==('cam_on','S07','SES01','cam_on_S07_SES01_002')
    assert check_trial_layout(b,mb,root)==('cam_on','S07','SES01','wrong_position')
    rs=manifest_rows(root)
    assert [x['condition'] for x in rs]==['correct','wrong_position']
    assert all(x['session']=='SES01' for x in rs)
    assert rs[1]['sample_id']=='cam_on_S07_SES01_002'


def test_new_subject_and_session_separate_counters(tmp_path):
    root=tmp_path/'feature_diagnostic'
    create(root,session='SES01')
    other,_=create(root,session='SES02')
    third,_=create(root,subject='S08')
    assert other.name.endswith('SES02_001')
    assert third.name.endswith('SES01_001')


def test_numbering_skips_existing_and_works_after_rebuild(tmp_path):
    root=tmp_path/'feature_diagnostic'
    for n in ('001','003','006'):
        (root/'cam_on'/'S07'/'SES01'/f'cam_on_S07_SES01_{n}').mkdir(parents=True)
    dest,sid=next_sample_path(root,'cam_on','S07','SES01')
    assert sid=='cam_on_S07_SES01_007'


def test_rebuild_recognizes_new_and_legacy_without_duplication(tmp_path):
    root=tmp_path/'feature_diagnostic'
    new,meta=create(root)
    legacy=root/'cam_on'/'S01'/'partial'/'20261027T111111Z'
    legacy.mkdir(parents=True)
    np.savez_compressed(legacy/'raw_landmarks.npz',dummy=[1])
    np.savez_compressed(legacy/'features.npz',features453=[[1]])
    oldmeta={'sample_id':legacy.name,'label':'cam_on','subject':'S01',
             'condition':'partial','session':'SES02','frames':1}
    write_metadata(legacy,oldmeta)
    a=rebuild_trial_manifest(root)
    assert a['scanned']==2 and a['total']==2 and not a['skipped']
    assert len(collect_rows(root,label='cam_on',deep_video_check=False))==2
    assert manifest_rows(root)[1]['session']=='SES02'


def test_reviewer_filters_session_and_uses_metadata_condition(tmp_path):
    root=tmp_path/'feature_diagnostic'
    create(root,session='SES01',condition='wrong_position')
    create(root,session='SES02',condition='wrong_handshape')
    rows=collect_rows(root,label='cam_on',session='SES02',deep_video_check=False)
    assert len(rows)==1
    assert rows[0]['condition']=='wrong_handshape'
    assert rows[0]['sample_id']=='cam_on_S07_SES02_001'


def test_layout_rejects_wrong_subject_or_sample_id(tmp_path):
    root=tmp_path/'feature_diagnostic'
    trial,meta=create(root)
    mismatch=dict(meta,subject='S09')
    with pytest.raises(ValueError,match='mismatch'):
        check_trial_layout(trial,mismatch,root)
    mismatch=dict(meta,sample_id='cam_on_S07_SES01_999')
    with pytest.raises(ValueError,match='mismatch'):
        check_trial_layout(trial,mismatch,root)


def test_migration_dry_run_and_explicit_move(tmp_path):
    root=tmp_path/'feature_diagnostic'
    legacy=root/'cam_on'/'S07'/'wrong_handshape'/'20261028T071234Z'
    legacy.mkdir(parents=True)
    np.savez_compressed(legacy/'raw_landmarks.npz',dummy=[1])
    np.savez_compressed(legacy/'features.npz',dummy=[2])
    write_metadata(legacy,{'sample_id':legacy.name,'subject':'S07','label':'cam_on',
                            'session':'SES01','condition':'wrong_handshape','frames':1})
    raw_bytes=(legacy/'raw_landmarks.npz').read_bytes()
    actions,skipped=plan(root)
    assert len(actions)==1 and not skipped
    assert legacy.is_dir()  # dry-run didn't move anything
    result=migrate(root,actions)
    new=root/'cam_on'/'S07'/'SES01'/'cam_on_S07_SES01_001'
    assert result['manifest_result']['total']==1
    assert not legacy.exists() and new.is_dir()
    assert (new/'raw_landmarks.npz').read_bytes()==raw_bytes
    newmeta=json.loads((new/'metadata.json').read_text())
    assert newmeta['legacy_sample_id']=='20261028T071234Z'
    assert newmeta['condition']=='wrong_handshape'
    assert manifest_rows(root)[0]['sample_id']==new.name


def test_does_not_guess_missing_legacy_session(tmp_path):
    root=tmp_path/'feature_diagnostic'
    legacy=root/'cam_on'/'S07'/'wrong_position'/'20261028T071234Z'
    legacy.mkdir(parents=True)
    write_metadata(legacy,{'sample_id':legacy.name,'label':'cam_on','subject':'S07',
                           'condition':'wrong_position'})
    actions,skipped=plan(root)
    assert not actions and skipped and skipped[0]['reason'].startswith('NO_SESSION')


def test_video_and_landmarks_share_new_session_sample_folder(tmp_path):
    from core.feature_diagnostic_video import save_source_video
    root=tmp_path/'feature_diagnostic'
    trial,meta=create(root,condition='correct')
    frames=[]
    for i in range(8):
        frame=np.zeros((90,120,3),np.uint8)
        frame[:, :, 0]=i*20
        frames.append(frame)
    video=save_source_video(trial,frames,[i/25.0 for i in range(8)])
    meta['frames']=8
    meta['video']=video
    write_metadata(trial,meta)
    append_trial_manifest(root,trial,meta)
    row=collect_rows(root,label='cam_on',subject='S07',session='SES01')[0]
    assert row['status']=='PASS'
    assert row['video_frames']==8
    assert row['sample_id']=='cam_on_S07_SES01_001'
    assert (trial/'source_video.avi').is_file()
    assert Path(row['raw_path']).is_file()
    assert Path(row['features_path']).is_file()
    assert manifest_rows(root)[0]['video_status']=='SAVED_VERIFIED'


def test_spatial_comparison_reads_session_folder_by_metadata(tmp_path):
    from core.spatial_handshape_diagnostic import pairwise_report
    root=tmp_path/'feature_diagnostic'
    a,ma=create(root,condition='correct')
    b,mb=create(root,condition='wrong_position')
    for path,value in ((a,0.1),(b,0.5)):
        np.savez_compressed(path/'features.npz',
                            features453=np.full((16,453),value,np.float32),
                            candidate_v4_32=np.full((16,32),value,np.float32),
                            candidate_v4_mask32=np.ones((16,32),bool))
    records,report=pairwise_report(root,'cam_on','S07',root/'reports')
    assert report.is_file()
    assert len(records)==11
    assert {r['condition'] for r in records}=={'wrong_position'}
