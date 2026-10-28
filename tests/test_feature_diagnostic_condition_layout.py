from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from core.feature_diagnostic_video import (
    next_sample_path, check_trial_layout, append_trial_manifest,
    rebuild_trial_manifest, write_metadata, probe_video, save_source_video,
    DIAGNOSTIC_CONDITIONS,
)
from scripts.review_feature_diagnostic_videos import collect_rows, render_contact_sheet
from scripts.migrate_feature_diagnostic_layout import plan, migrate
from scripts.live_feature_diagnostic import parse_args
from core.spatial_handshape_diagnostic import pairwise_report


def create(root, label='xin_chao', subject='S03', session='SES01', condition='partial',
           video=False):
    dest, sid = next_sample_path(root, label, subject, session, condition)
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'raw_landmarks.npz', dummy=np.ones((32,21,3),np.float32))
    np.savez_compressed(dest/'features.npz', features453=np.ones((32,453),np.float32))
    meta = {'sample_id':sid, 'label':label, 'subject':subject, 'subject_id':subject,
            'session':session, 'session_id':session, 'condition':condition,
            'frames':32, 'captured_at':'2026-10-28T08:00:00Z',
            'qc':{'any_hand_ratio':.95,'pose_ratio':.99}}
    if video:
        frames=[np.full((96,128,3),i*4,np.uint8) for i in range(32)]
        meta['video']=save_source_video(dest,frames,[i/25.0 for i in range(32)])
    else:
        meta['video']={'status':'DISABLED'}
    write_metadata(dest,meta)
    append_trial_manifest(root,dest,meta)
    return dest,meta


def manifest_rows(root):
    with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def test_exact_uploaded_screenshot_layout(tmp_path):
    root=tmp_path/'feature_diagnostic'
    trial,meta=create(root)
    assert trial.relative_to(root).parts == ('xin_chao','S03','SES01','partial','partial_S03_SES01_001')
    assert check_trial_layout(trial,meta,root)==('xin_chao','S03','SES01','partial')


def test_six_conditions_have_independent_counters(tmp_path):
    root=tmp_path/'f'
    for cond in DIAGNOSTIC_CONDITIONS:
        a,_=create(root,condition=cond)
        b,_=create(root,condition=cond)
        assert a.name==f'{cond}_S03_SES01_001'
        assert b.name==f'{cond}_S03_SES01_002'
    assert len(manifest_rows(root))==12


def test_new_session_separate_counter(tmp_path):
    root=tmp_path/'f'
    create(root)
    b,_=create(root,session='SES02')
    assert b.name=='partial_S03_SES02_001'


def test_new_signer_separate_counter(tmp_path):
    root=tmp_path/'f'
    create(root)
    b,_=create(root,subject='S07')
    assert b.name=='partial_S07_SES01_001'


def test_next_sample_skips_gaps_without_overwrite(tmp_path):
    root=tmp_path/'f'
    for n in ('001','003','010'):
        (root/'xin_chao'/'S03'/'SES01'/'partial'/f'partial_S03_SES01_{n}').mkdir(parents=True)
    p,sid=next_sample_path(root,'xin_chao','S03','SES01','partial')
    assert sid=='partial_S03_SES01_011'
    assert not p.exists()


def test_reject_invalid_conditions(tmp_path):
    with pytest.raises(ValueError,match='condition'):
        next_sample_path(tmp_path,'xin_chao','S03','SES01','fake')


def test_metadata_condition_and_session_verified(tmp_path):
    root=tmp_path/'f'
    p,m=create(root)
    with pytest.raises(ValueError):
        check_trial_layout(p,{**m,'condition':'wrong_position'},root)
    with pytest.raises(ValueError):
        check_trial_layout(p,{**m,'session':'SES99','session_id':'SES99'},root)
    with pytest.raises(ValueError):
        check_trial_layout(p,{**m,'subject':'S02'},root)
    with pytest.raises(ValueError):
        check_trial_layout(p,{**m,'sample_id':'partial_S03_SES01_999'},root)


def test_session_cli_no_signer_alias():
    a=parse_args(['--label','xin_chao','--subject','S03','--session','SES01','--frames','32'])
    assert a.session=='SES01' and a.save_video
    assert not hasattr(a,'signer_alias')
    with pytest.raises(SystemExit):
        parse_args(['--signer-alias','Some name'])


def test_raw_features_and_video_copackaged(tmp_path):
    root=tmp_path/'f'
    p,meta=create(root,video=True)
    assert sorted(x.name for x in p.iterdir())==[
        'features.npz','metadata.json','raw_landmarks.npz','source_video.avi']
    assert probe_video(p/'source_video.avi',expected_frames=32)['decoded_frames']==32
    assert len(manifest_rows(root))==1
    assert manifest_rows(root)[0]['condition']=='partial'
    assert manifest_rows(root)[0]['session']=='SES01'
    assert manifest_rows(root)[0]['video_status']=='SAVED_VERIFIED'


def test_manifest_upsert_and_rebuild_idempotent(tmp_path):
    root=tmp_path/'f'
    p,m=create(root)
    append_trial_manifest(root,p,m)
    assert len(manifest_rows(root))==1
    assert rebuild_trial_manifest(root)['scanned']==1
    assert rebuild_trial_manifest(root)['scanned']==1
    assert len(manifest_rows(root))==1


def test_reviewer_reads_new_session_condition_layout(tmp_path):
    root=tmp_path/'f'
    create(root,video=True)
    create(root,condition='wrong_position')
    create(root,session='SES02',condition='wrong_handshape')
    rows=collect_rows(root, label='xin_chao',subject='S03',session='SES01',deep_video_check=True)
    assert len(rows)==2
    assert set(r['condition'] for r in rows)=={'partial','wrong_position'}
    rows=collect_rows(root,condition='partial',deep_video_check=True)
    assert len(rows)==1 and rows[0]['status']=='PASS'
    target=tmp_path/'sheet.jpg'
    assert render_contact_sheet(rows[0],target)
    assert target.stat().st_size>0


def make_old(root, path_parts, session, cond):
    old=root.joinpath(*path_parts)
    old.mkdir(parents=True)
    np.savez_compressed(old/'raw_landmarks.npz',dummy=np.ones(2))
    np.savez_compressed(old/'features.npz',dummy=np.ones(2))
    meta={'label':'xin_chao','subject':'S03','session':session,
          'condition':cond,'sample_id':old.name,'frames':2}
    write_metadata(old,meta)
    return old


def test_migrate_intermediate_session_layout(tmp_path):
    root=tmp_path/'f'
    old=make_old(root,('xin_chao','S03','SES01','xin_chao_S03_SES01_001'),
                 'SES01','partial')
    old_data=(old/'raw_landmarks.npz').read_bytes()
    acts,skips=plan(root)
    assert len(acts)==1 and not skips
    assert old.exists()
    res=migrate(root,acts)
    dest=root/'xin_chao'/'S03'/'SES01'/'partial'/'partial_S03_SES01_001'
    assert res['manifest_result']['total']==1
    assert not old.exists() and dest.exists()
    assert (dest/'raw_landmarks.npz').read_bytes()==old_data
    new_meta=json.loads((dest/'metadata.json').read_text())
    assert new_meta['legacy_sample_id']=='xin_chao_S03_SES01_001'
    assert new_meta['sample_id']=='partial_S03_SES01_001'
    assert manifest_rows(root)[0]['trial_dir']==str(dest)
    assert not plan(root)[0]


def test_migrate_original_condition_timestamp_layout(tmp_path):
    root=tmp_path/'f'
    old=make_old(root,('xin_chao','S03','wrong_handshape','20261028T101010Z'),
                 'SES02','wrong_handshape')
    actions,skipped=plan(root)
    assert len(actions)==1 and not skipped
    migrate(root,actions)
    new=root/'xin_chao'/'S03'/'SES02'/'wrong_handshape'/'wrong_handshape_S03_SES02_001'
    assert new.is_dir() and not old.exists()
    assert check_trial_layout(new,json.loads((new/'metadata.json').read_text()),root)[-1]=='wrong_handshape'


def test_migrate_no_session_requires_explicit_default(tmp_path):
    root=tmp_path/'f'
    old=make_old(root,('xin_chao','S03','partial','20261028T101010Z'), '', 'partial')
    actions,skips=plan(root)
    assert not actions and 'NO_SESSION' in skips[0]['reason']
    actions,skips=plan(root,'SES99')
    assert len(actions)==1
    assert not skips
    assert old.exists()  # only dry run


def test_migrate_distributes_conditions_separately(tmp_path):
    root=tmp_path/'f'
    make_old(root,('xin_chao','S03','SES01','xin_chao_S03_SES01_001'), 'SES01','correct')
    make_old(root,('xin_chao','S03','SES01','xin_chao_S03_SES01_002'), 'SES01','partial')
    actions,skips=plan(root)
    assert len(actions)==2 and not skips
    migrate(root,actions)
    assert (root/'xin_chao'/'S03'/'SES01'/'correct'/'correct_S03_SES01_001').exists()
    assert (root/'xin_chao'/'S03'/'SES01'/'partial'/'partial_S03_SES01_001').exists()
    assert len(manifest_rows(root))==2


def test_migrate_does_not_overwrite_existing(tmp_path):
    root=tmp_path/'f'
    create(root,condition='partial')
    make_old(root,('xin_chao','S03','SES01','xin_chao_S03_SES01_004'), 'SES01','partial')
    actions,_=plan(root)
    assert actions[0]['new_id']=='partial_S03_SES01_002'
    migrate(root,actions)
    assert (root/'xin_chao'/'S03'/'SES01'/'partial'/'partial_S03_SES01_001').is_dir()
    assert (root/'xin_chao'/'S03'/'SES01'/'partial'/'partial_S03_SES01_002').is_dir()


def test_migrate_preflight_prevents_changes(tmp_path):
    root=tmp_path/'f'
    old=make_old(root,('xin_chao','S03','SES01','xin_chao_S03_SES01_001'), 'SES01','correct')
    actions,_=plan(root)
    dest=Path(actions[0]['dest'])
    dest.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='preflight'):
        migrate(root,actions)
    assert old.exists()


def test_rebuild_both_old_and_new_layouts(tmp_path):
    root=tmp_path/'f'
    create(root)
    old=make_old(root,('xin_chao','S03','wrong_position','20261028T111111Z'), 'SES02','wrong_position')
    old2=make_old(root,('xin_chao','S03','SES03','xin_chao_S03_SES03_004'), 'SES03','wrong_handshape')
    result=rebuild_trial_manifest(root)
    assert result['scanned']==3 and result['total']==3
    assert len(collect_rows(root,deep_video_check=False))==3


def test_pairwise_report_reads_condition_subfolders(tmp_path):
    root=tmp_path/'f'
    a,_=create(root,condition='correct')
    b,_=create(root,condition='wrong_position')
    for p,value in ((a,0.1),(b,0.4)):
        np.savez_compressed(p/'features.npz',
            features453=np.full((32,453),value,np.float32),
            candidate_v4_32=np.full((32,32),value,np.float32),
            candidate_v4_mask32=np.ones((32,32),bool))
    rs,fp=pairwise_report(root,'xin_chao','S03',root/'reports')
    assert len(rs)==11 and fp.is_file()
    assert {r['condition'] for r in rs}=={'wrong_position'}


def test_manifest_keeps_all_condition_rows(tmp_path):
    root=tmp_path/'f'
    for c in DIAGNOSTIC_CONDITIONS:
        create(root,condition=c)
    result=rebuild_trial_manifest(root)
    assert result['total']==6
    assert {r['condition'] for r in manifest_rows(root)}==set(DIAGNOSTIC_CONDITIONS)
    assert all(r['sample_id'].startswith(r['condition']+'_') for r in manifest_rows(root))


def test_missing_npz_is_skipped_from_rebuild(tmp_path):
    root=tmp_path/'f'
    p,m=create(root)
    (p/'features.npz').unlink()
    res=rebuild_trial_manifest(root)
    assert res['total']==0 and res['skipped']


def test_no_session_alias_column(tmp_path):
    root=tmp_path/'f'
    create(root)
    assert 'signer_alias' not in manifest_rows(root)[0]


def test_layout_audit_passes_complete_new_samples(tmp_path):
    from scripts.audit_feature_diagnostic_layout import audit
    root=tmp_path/'f'
    create(root,condition='partial',video=True)
    create(root,condition='correct',video=True)
    report=audit(root)
    assert report['passed'] and report['canonical_samples']==2
    assert report['counts_by_condition']=={'correct':1,'partial':1}


def test_layout_audit_flags_missing_video_and_stale_manifest(tmp_path):
    from scripts.audit_feature_diagnostic_layout import audit
    root=tmp_path/'f'
    p,m=create(root,condition='partial',video=True)
    (p/'source_video.avi').unlink()
    result=audit(root)
    assert not result['passed'] and result['status_counts']['REVIEW']==1
    assert 'MISSING_VIDEO:source_video.avi' in result['records'][0]['errors']
