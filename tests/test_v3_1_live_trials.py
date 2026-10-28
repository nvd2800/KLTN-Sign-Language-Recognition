from scripts.analyze_v3_realtime_trials import score_trials


def test_marked_negative_trial_with_false_token():
    rows=[
        {'trial_id':'1','source':'TRIAL_START','expected_label':'UNKNOWN','test_condition':'wrong_position','time_s':'1'},
        {'trial_id':'1','source':'TCN_V3_SPATIAL','label':'cam_on','fsm':'STABLE','emitted_token':'cam_on'},
        {'trial_id':'1','source':'NO_HAND_RELEASE_HEURISTIC'},
        {'trial_id':'1','source':'TRIAL_END','time_s':'3'},
        {'trial_id':'2','source':'TRIAL_START','expected_label':'xin_chao','test_condition':'correct','time_s':'4'},
        {'trial_id':'2','source':'TCN_V3_SPATIAL','label':'xin_chao','fsm':'WAIT_RELEASE','emitted_token':'xin_chao'},
        {'trial_id':'2','source':'MANUAL_REST'},
        {'trial_id':'2','source':'TRIAL_END','time_s':'7'}
    ]
    r=score_trials(rows)
    assert r['n_complete_negative_trials']==1 and r['false_token_negative_trials']==1
    assert r['n_complete_correct_trials']==1 and r['correct_token_trials']==1
    assert r['release_source_counts']=={'NO_HAND_RELEASE_HEURISTIC':1,'MANUAL_REST':1}


def test_incomplete_trial_excluded_from_rate():
    r=score_trials([{'trial_id':'8','source':'TRIAL_START','test_condition':'wrong_handshape','expected_label':'UNKNOWN','time_s':'3'}])
    assert r['n_complete_negative_trials']==0 and r['trial_false_token_rate'] is None
