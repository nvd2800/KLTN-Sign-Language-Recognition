"""PHASE 09 — Continuous webcam recognition with trained 5-class TCN V3/V3.1 Spatial.

Copy alongside previous Phase08/Phase09 code and run from project root:
  python scripts/live_phase09_tcn_v3.py --fullscreen

Uses existing Phase04/05/06, Phase07 resampling, V3 TRAIN-only scaler,
and Phase09 state machine. Does NOT train or overwrite the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_phase4 import load_phase4_config
from core.feature_phase5 import load_phase5_config
from core.feature_phase6 import load_phase6_config
from core.phase8_live import FrameSample, build_model_input, append_csv_row
from core.continuous_phase9 import ContinuousRecognizer, Phase09Config, State
from core.release_v3 import NoHandReleaseScheduler
from core.phase9_live_logic import (
    LiveGateConfig, WindowDecision, decide_window,
    readable_tokens,
)


def at_root(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_args():
    p = argparse.ArgumentParser(description="Phase 09 continuous webcam — TCN V3/V3.1 Spatial UNKNOWN")
    p.add_argument("--config", default="configs/phase9_v3_spatial.yaml")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--cam-width", type=int, default=None)
    p.add_argument("--cam-height", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="Infer once every N new frames")
    p.add_argument("--window-frames", type=int, default=None, help="Length of sliding camera window")
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--min-margin", type=float, default=None)
    p.add_argument("--min-motion", type=float, default=None, help="0 = motion gate OFF")
    p.add_argument("--no-auto-release", action="store_true", help="Only TCN REST or manual B releases")
    p.add_argument("--spatial-gate", default=None, help="Enable spatial gate with profile.json")
    p.add_argument("--require-validated-gate", action="store_true", help="Reject DEV_ONLY gate profiles")
    p.add_argument('--target-label', default='xin_chao', help='Reference sign for labelled trials 1..5')
    p.add_argument('--trial-isolation', action='store_true',
                   help='Reset FSM at every labelled trial, for isolated trials only; OFF for continuous testing')
    p.add_argument("--phase10-config",default=None,help="Enable optional Phase10 Sentence+TTS (YAML)")
    return p.parse_args()


def as_xyz(landmarks, n):
    if landmarks is None or len(landmarks.landmark) < n:
        return np.zeros((n, 3), np.float32), False
    a = np.asarray([(x.x, x.y, x.z) for x in landmarks.landmark[:n]], dtype=np.float32)
    if a.shape != (n, 3) or not np.isfinite(a).all():
        return np.zeros((n, 3), np.float32), False
    return a, True


def from_holistic(result, timestamp):
    l,lp=as_xyz(result.left_hand_landmarks,21)
    r,rp=as_xyz(result.right_hand_landmarks,21)
    p,pp=as_xyz(result.pose_landmarks,33)
    f,fp=as_xyz(result.face_landmarks,468)
    return FrameSample(l,r,p,f,lp,rp,pp,fp,float(timestamp))


def draw_landmarks(image, res, mp, drawing):
    blue = drawing.DrawingSpec(color=(250,115,30), thickness=3, circle_radius=3)
    orange = drawing.DrawingSpec(color=(0,150,255), thickness=3, circle_radius=3)
    green = drawing.DrawingSpec(color=(20,220,80), thickness=2, circle_radius=2)
    violet = drawing.DrawingSpec(color=(210,100,230), thickness=1, circle_radius=1)
    if res.pose_landmarks is not None:
        drawing.draw_landmarks(image, res.pose_landmarks, mp.POSE_CONNECTIONS, green, green)
    if res.face_landmarks is not None:
        drawing.draw_landmarks(image, res.face_landmarks, mp.FACEMESH_CONTOURS, None, violet)
    if res.left_hand_landmarks is not None:
        drawing.draw_landmarks(image, res.left_hand_landmarks, mp.HAND_CONNECTIONS, blue, blue)
    if res.right_hand_landmarks is not None:
        drawing.draw_landmarks(image, res.right_hand_landmarks, mp.HAND_CONNECTIONS, orange, orange)
    return image


def fit_contain(frame, width, height):
    """Preserve full webcam frame without cropping."""
    h,w=frame.shape[:2]
    k=min(width/max(1,w),height/max(1,h))
    nw=max(1,int(round(w*k))); nh=max(1,int(round(h*k)))
    scaled=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    canvas=np.full((height,width,3),(12,17,24),np.uint8)
    x=(width-nw)//2;y=(height-nh)//2
    canvas[y:y+nh,x:x+nw]=scaled
    return canvas


def txt(im, message, pos, color=(238,238,242), scale=.53, bold=False):
    cv2.putText(im,str(message),(int(pos[0]),int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX,scale,color,2 if bold else 1,cv2.LINE_AA)


def render(frame, labels, latest, engine, fps, infer_ms, flags, width, height, gate, tick, spatial_last=None):
    left_width=max(480,int(width*.69))
    left_width=min(left_width,width-360)
    view=fit_contain(frame,left_width,height)
    canvas=np.full((height,width,3),(28,33,44),np.uint8)
    canvas[:,:left_width]=view
    sidebar_x=left_width+18
    x=sidebar_x
    w=width
    cv2.line(canvas,(left_width,0),(left_width,height),(90,100,115),2)
    y=34
    txt(canvas,"PHASE 09 | TCN V3/V3.1 Spatial",(x,y),(230,232,250),.65,True);y+=30
    txt(canvas,"5-class +60D spatial recognition",(x,y),(172,188,210),.46);y+=29
    txt(canvas,f"FSM: {engine.state.value}",(x,y),(0,210,250),.63,True);y+=28
    txt(canvas,f"FPS {fps:.1f}  |  inference {infer_ms:.0f} ms",(x,y),(170,195,209),.46);y+=28
    txt(canvas,f"Buffer {flags['buffer']}/{flags['window']}, stride {flags['stride']}",(x,y),(175,191,211),.46);y+=25
    txt(canvas,f"L:{'ON' if flags['left'] else 'OFF'} R:{'ON' if flags['right'] else 'OFF'} "
               f"P:{'ON' if flags['pose'] else 'OFF'} F:{'ON' if flags['face'] else 'OFF'}",
        (x,y),(190,213,213),.43);y+=23
    txt(canvas,f"No hands {flags['no_hands']} frames | auto release: "
               f"{'ON' if flags['auto_release'] else 'OFF'}",(x,y),(198,209,211),.44);y+=24
    txt(canvas,f"Conf >= {gate.min_confidence:.2f} (EXPERIMENTAL)",(x,y),(10,195,255),.45);y+=30
    cv2.line(canvas,(x,y),(w-15,y),(90,100,115),1);y+=24

    if latest is None:
        txt(canvas,"PREDICTION: waiting...",(x,y),(170,175,188),.50);y+=25
    else:
        txt(canvas,f"RAW: {latest.raw_label}  {latest.confidence*100:.1f}%",
            (x,y),(210,210,228),.53,True);y+=30
        c=(0,218,100) if latest.label not in ("UNKNOWN","REST") else (75,180,245)
        txt(canvas,f"DECISION: {latest.label}",(x,y),c,.64,True);y+=29
        txt(canvas,f"Reason: {latest.reason}",(x,y),(182,189,205),.43);y+=24
        txt(canvas,f"Margin: {latest.margin:.3f} | hand: {latest.presence_ratio:.2f}",
            (x,y),(171,184,200),.44);y+=23
        for label,p in latest.probabilities.items():
            bar_x=x+112;bar_w=max(70,w-25-bar_x-50)
            txt(canvas,label[:15],(x,y),(210,215,218),.42)
            cv2.rectangle(canvas,(bar_x,y-13),(bar_x+bar_w,y),(74,80,92),-1)
            cv2.rectangle(canvas,(bar_x,y-13),(bar_x+int(bar_w*min(1,max(0,p))),y),(35,182,232),-1)
            txt(canvas,f"{p*100:.0f}%",(bar_x+bar_w+5,y),(195,215,218),.39)
            y+=24
    y+=10
    if spatial_last is not None:
        spatial_color = (50,230,130) if spatial_last.decision=="PASS" else (45,120,255)
        txt(canvas, "SPATIAL: " + spatial_last.decision[:21], (x,y), spatial_color,.46,True);y+=25
        txt(canvas,spatial_last.reason[:40],(x,y),(160,185,216),.37);y+=22
    cv2.line(canvas,(x,y),(w-15,y),(90,100,115),1);y+=25
    snap=engine.snapshot()
    txt(canvas,f"Candidate: {snap['candidate'] or '-'} ({snap['candidate_count']})",
        (x,y),(203,220,225),.48);y+=27
    txt(canvas,f"Release: {snap['release_count']}/{engine.cfg.release_consecutive}",
        (x,y),(203,220,225),.48);y+=25
    txt(canvas,f"CONFIRMED TOKENS: {len(engine.events)}",(x,y),(80,230,130),.55,True);y+=28
    for t in snap['tokens'][-8:]:
        txt(canvas,f"> {t}",(x,y),(105,235,156),.50,True);y+=25
        if y>=height-175:break
    cv2.rectangle(canvas,(left_width,height-150),(width,height),(18,22,30),-1)
    for i,line in enumerate([
        "F full | V landmarks | M mirror",
        "P pause | C clear tokens | R reset buffer",
        "B manual REST (debug) | S snapshot",
        "0-6 trial labels | A auto release | Q quit",
    ]):
        txt(canvas,line,(x,height-121+i*26),(206,210,224),.40)
    # Text across camera bottom; not cropped.
    txt(canvas,"CONFIRMED: " + readable_tokens(snap['tokens'],6),
        (15,height-28),(100,244,165),.65,True)
    if not flags['running']:
        txt(canvas,"PAUSED — P to resume",(22,55),(30,140,255),.75,True)
    if tick:
        txt(canvas,"TOKEN CONFIRMED",(22,96),(25,242,85),.82,True)
    return canvas


CSV_FIELDS = [
    "timestamp_utc","time_s","source","raw_label","label","confidence","margin",
    "reason","hand_presence_ratio","window_motion","fsm","candidate","candidate_count",
    "release_count","emitted_token","inference_ms","probs_json",
    "spatial_decision","spatial_reason","spatial_scores_json","spatial_thresholds_json",
    "trial_id","test_condition","expected_label","target_sign",
]


def main():
    a=parse_args()
    cfg=yaml.safe_load(at_root(a.config).read_text(encoding='utf-8'))
    paths=cfg['artifacts']; camera_cfg=cfg['camera']; icfg=cfg['inference']; rcfg=cfg['recognition']; uicfg=cfg['ui']; logs=cfg['logging']
    sg_cfg = cfg.get("spatial_gate", {})
    sg_path = a.spatial_gate or (sg_cfg.get("profile") if sg_cfg.get("enabled",False) else None)
    if a.require_validated_gate and not sg_path:
        raise RuntimeError("--require-validated-gate requires an enabled spatial gate profile")
    spatial_profile = None
    if sg_path:
        # Legacy 32D gate is optional and loaded only when explicitly requested.
        from core.spatial_handshape_diagnostic import v4_candidate32
        from core.spatial_validation_gate import load_profile, evaluate as evaluate_spatial_gate
        spatial_profile = load_profile(at_root(sg_path))
    if spatial_profile:
        if a.require_validated_gate and spatial_profile["calibration_mode"] != "VALIDATED":
            raise RuntimeError("Profile is DEV_ONLY; --require-validated-gate requires VALIDATED")
        print("[SPATIAL GATE] Enabled:",at_root(sg_path),"mode",spatial_profile["calibration_mode"])
        print("[SPATIAL GATE] Protected classes:",list(spatial_profile["classes"]))
        if spatial_profile["calibration_mode"] == "DEV_ONLY":
            print("WARNING: DEV_ONLY threshold may overfit your diagnostic clips; not a validated threshold!")
        print("WARNING: Unconfigured classes BYPASS spatial checks. UI/logs mark BYPASS_UNCONFIGURED.")
    model_path,stats_path,labels_path=(at_root(paths[k]) for k in ('model','stats','labels'))
    for p in (model_path,stats_path,labels_path):
        if not p.is_file():
            raise FileNotFoundError(f"TCN V3/V3.1 Spatial artifact not found: {p}. Train V3 Spatial first.")
    with labels_path.open(encoding='utf-8') as f:
        labels_obj=json.load(f)
    labels=labels_obj['labels'] if isinstance(labels_obj,dict) else labels_obj
    if not isinstance(labels,list) or len(labels)!=5 or 'REST' not in labels or 'UNKNOWN' not in labels or len(set(labels))!=5:
        raise RuntimeError(f"TCN V3/V3.1 Spatial requires 5 unique classes including REST and UNKNOWN: {labels}")
    with np.load(stats_path,allow_pickle=False) as d:
        mean=np.asarray(d['mean'],np.float32)
        std=np.asarray(d['std'],np.float32)
        target=int(np.asarray(d['target_frames']).reshape(-1)[0])
        feature_dim=int(np.asarray(d['feature_dim']).reshape(-1)[0])
    if (target,feature_dim)!=(32,513) or mean.shape!=(513,) or std.shape!=(513,):
        raise RuntimeError(f"TCN V3/V3.1 Spatial preprocessing mismatch: target={target}, dim={feature_dim}")
    cfg4,_=load_phase4_config(at_root('configs/feature_phase4.yaml'))
    cfg5,_=load_phase5_config(at_root('configs/feature_phase5.yaml'))
    cfg6,_=load_phase6_config(at_root('configs/feature_phase6.yaml'))
    gate=LiveGateConfig(
        min_confidence=float(a.min_confidence if a.min_confidence is not None else icfg.get('min_confidence',.80)),
        min_margin=float(a.min_margin if a.min_margin is not None else icfg.get('min_margin',0)),
        min_hand_presence_ratio=float(icfg.get('min_hand_presence_ratio',.30)),
        min_motion=float(a.min_motion if a.min_motion is not None else icfg.get('min_motion',0)),
        release_no_hand_frames=int(icfg.get('release_no_hand_frames',12)),
        auto_release_no_hand=bool(icfg.get('auto_release_no_hand',True) and not a.no_auto_release),
    )
    gate.validate()
    min_count=int(rcfg['min_consecutive']);rel_count=int(rcfg['release_consecutive'])
    engine=ContinuousRecognizer(Phase09Config(
        confidence_threshold=gate.min_confidence, min_consecutive=min_count,
        release_consecutive=rel_count,
        min_token_interval_s=float(rcfg['min_token_interval_s']),
        max_gap_s=float(rcfg['max_gap_s']), rest_label='REST', unknown_label='UNKNOWN',
        unknown_counts_as_release=False,
    ))
    phase10_builder=phase10_store=phase10_speaker=phase10_voice=None
    phase10_auto_speak=False
    phase10_banner=True
    if a.phase10_config:
        from core.phase10_sentence import SentenceBuilder
        from core.phase10_store import Phase10Store
        from core.phase10_tts import VietnameseSpeaker, SpeechQueue
        from core.phase10_ui import draw_phase10_banner
        p10=yaml.safe_load(at_root(a.phase10_config).read_text(encoding='utf-8'))
        phase10_builder=SentenceBuilder(p10.get('phrases'),max_tokens=int(p10.get('sentence',{}).get('max_tokens',40)), punctuation_labels=p10.get('punctuation_labels'))
        phase10_store=Phase10Store(at_root(p10.get('logging',{}).get('root','logs/phase10_sessions')))
        p10_tts=p10.get('tts',{})
        phase10_voice=VietnameseSpeaker(
            p10_tts.get('backend','auto'),rate=int(p10_tts.get('rate',145)),
            model_path=at_root(p10_tts.get('model_path','voices/piper/vi_VN-vais1000-medium.onnx')),
            wav_output_dir=at_root(p10_tts.get('wav_output_dir','logs/phase10_audio')),
            play_audio=bool(p10_tts.get('play_audio',True)),
            vieneu_endpoint=p10_tts.get('vieneu_endpoint','http://127.0.0.1:8770/synthesize'),
            vieneu_voice=p10_tts.get('vieneu_voice','Minh Đức'))
        phase10_speaker=SpeechQueue(phase10_voice)
        phase10_auto_speak=bool(p10.get('sentence',{}).get('auto_speak_on_finalize',False))
        phase10_banner=bool(p10.get('ui',{}).get('show_banner',True))
        phase10_store.append('SESSION_START',phase09_config=str(at_root(a.config)),model_path=str(model_path),
                             feature_shape=[32,513],tts_backend=phase10_voice.info.backend,
                             tts_available=phase10_voice.info.available)
        print('[PHASE10] Enabled | session:',phase10_store.session_dir)
        print('[PHASE10] Vietnamese TTS:',phase10_voice.info.detail)
        print('[PHASE10] Keys: N or Space=finish sentence | Z=undo | X=clear pending | T=speak | Y=auto speak | E=export')

    win_frames=int(a.window_frames or icfg['window_frames'])
    stride=int(a.stride or icfg['stride_frames'])
    if win_frames < 2 or stride < 1:
        raise ValueError('window_frames >= 2 and stride >= 1 required; fewer than 32 frames are zero-padded to model input length 32')

    import tensorflow as tf
    print('Loading TCN V3/V3.1 Spatial:',model_path)
    print('Using scaler:', stats_path)
    print('Using labels:', labels_path)
    model=tf.keras.models.load_model(model_path,compile=False)
    shape=model.input_shape[0] if isinstance(model.input_shape,list) else model.input_shape
    if len(shape)!=3 or tuple(shape[1:])!=(32,513) or model.output_shape[-1]!=len(labels):
        raise RuntimeError(f'Incompatible model input/output: {model.input_shape} / {model.output_shape}')

    try:
        import mediapipe as mp
        holo=mp.solutions.holistic
        drawing=mp.solutions.drawing_utils
    except (ImportError,AttributeError) as exc:
        raise RuntimeError('MediaPipe Solutions unavailable; check mediapipe==0.10.21 and protobuf') from exc

    cam_idx=int(camera_cfg['id'] if a.camera is None else a.camera)
    cap=cv2.VideoCapture(cam_idx,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(cam_idx)
    if not cap.isOpened():cap=cv2.VideoCapture(cam_idx)
    if not cap.isOpened():raise RuntimeError('Webcam not found; try --camera 1 and close other camera apps')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(a.cam_width or camera_cfg['width']))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(a.cam_height or camera_cfg['height']))
    cap.set(cv2.CAP_PROP_FPS,float(camera_cfg['fps']))
    width=int(uicfg.get('width',1500));height=int(uicfg.get('height',880))
    fullscreen=bool(a.fullscreen or uicfg.get('fullscreen',False))
    win='Phase09 Continuous Recognition | TCN V3/V3.1 Spatial'
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win,width,height)
    if fullscreen:
        cv2.setWindowProperty(win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    buffer=deque(maxlen=win_frames)
    stride_counter=0;no_hand_frames=0
    latest=None;spatial_last=None;infer_ms=0.0;last_t=time.perf_counter();fps=0.0
    use_landmarks=bool(uicfg.get('show_landmarks',True))
    mirror=bool(camera_cfg.get('mirror_preview',True))
    running=True;last_event_time=0.0
    started=time.perf_counter()
    csv_path=at_root(logs['predictions']);token_path=at_root(logs['tokens'])
    release_scheduler=NoHandReleaseScheduler(
        threshold_frames=int(icfg.get('release_no_hand_frames',12)),
        interval_frames=int(icfg.get('release_interval_frames',2)))
    trial_id=0;test_condition='UNLABELED'
    session_tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    print('[TRIAL] 0 unlabeled | 1 correct | 2 wrong_position | 3 wrong_handshape | 4 wrong_orientation | 5 partial | 6 REST')
    snaps=at_root(logs['snapshots'])
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    token_path.parent.mkdir(parents=True,exist_ok=True)
    print('\nPHASE 09 LIVE – 5 CLASS TCN V3/V3.1 Spatial')
    print('Labels:',labels,'| gate:',gate.min_confidence,'| frame window:',win_frames,'| stride:',stride)
    print('WARNING: UNKNOWN rejection MUST be measured on independent signer TEST.')
    print('Keys: F fullscreen, V landmarks, M mirror, P pause, C clear, R reset buffer, B manual REST, A auto-release toggle, S snapshot, Q quit')

    def trial_marker(source:str,ts:float):
        # MARKERS are events, not TCN predictions and never create a token.
        append_csv_row(csv_path, {
            'timestamp_utc':datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'time_s':f'{ts-started:.4f}', 'source':source,
            'trial_id':f'{session_tag}:{trial_id}' if trial_id else '0','test_condition':test_condition,
            'target_sign':a.target_label,
            'expected_label': ('UNKNOWN' if test_condition.startswith('wrong_') or test_condition=='partial'
                               else ('REST' if test_condition=='REST'
                                     else (a.target_label if test_condition=='correct' else ''))),
            'fsm':engine.state.value,
        },CSV_FIELDS)

    def submit(label:str,conf:float,ts:float,source:str,raw:WindowDecision | None, ms=0.0):
        nonlocal last_event_time
        event=engine.update(label,conf,ts)
        snap=engine.snapshot()
        row={
            'timestamp_utc':datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'time_s':f'{ts-started:.4f}', 'source':source,
            'raw_label':raw.raw_label if raw else '', 'label':label,
            'confidence':f'{conf:.6f}', 'margin':f'{raw.margin:.6f}' if raw else '',
            'reason':raw.reason if raw else source,
            'hand_presence_ratio':f'{raw.presence_ratio:.4f}' if raw else '',
            'window_motion':f'{raw.motion:.5f}' if raw else '',
            'fsm':engine.state.value,
            'candidate':snap['candidate'] or '',
            'candidate_count':snap['candidate_count'], 'release_count':snap['release_count'],
            'emitted_token':event.label if event else '', 'inference_ms':f'{ms:.1f}',
            'probs_json':json.dumps(raw.probabilities) if raw else '',
            'spatial_decision':spatial_last.decision if raw and spatial_last else '',
            'spatial_reason':spatial_last.reason if raw and spatial_last else '',
            'spatial_scores_json':json.dumps(spatial_last.scores) if raw and spatial_last else '',
             'spatial_thresholds_json':json.dumps(spatial_last.thresholds) if raw and spatial_last else '',
            'trial_id':f'{session_tag}:{trial_id}' if trial_id else '0', 'test_condition':test_condition,
            'expected_label': ('UNKNOWN' if test_condition.startswith('wrong_') or test_condition=='partial'
                               else ('REST' if test_condition=='REST' else a.target_label if test_condition=='correct' else '')),
            'target_sign':a.target_label,
        }
        append_csv_row(csv_path,row,CSV_FIELDS)
        if event is not None:
            payload={
                'event':'TOKEN','label':event.label,
                'timestamp_s':round(event.timestamp_s-started,4),
                'confidence':round(event.confidence,6),
                'source':source,
                'model':str(model_path),
                'trial_id':f'{session_tag}:{trial_id}' if trial_id else '0','test_condition':test_condition,
                'total_tokens':len(engine.events),
                'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            }
            with token_path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(payload,ensure_ascii=False)+'\n')
            if phase10_builder is not None:
                if phase10_builder.add(event.label):
                    phase10_store.append('TOKEN_ACCEPTED',label=event.label,confidence=float(event.confidence),
                                         source=source,phase09_time_s=round(event.timestamp_s-started,4),
                                         preview=phase10_builder.preview)
                    print('[PHASE10 SENTENCE]',phase10_builder.preview)
                else:
                    phase10_store.append('IGNORED_LABEL',label=event.label,source=source)
            last_event_time=ts
            print(f"[TOKEN {len(engine.events):03d}] {event.label} conf={event.confidence:.3f} "
                  f"time={ts-started:.2f}s | {' '.join(snap['tokens'])}")
        return event

    def finalize_phase10(*, speak_if_enabled=True):
        if phase10_builder is None:
            return None
        sentence=phase10_builder.finalize()
        if sentence is None:
            return None
        phase10_store.sentence(sentence)
        phase10_store.export(phase10_builder)
        print('[PHASE10 FINALIZED]',sentence.text)
        if speak_if_enabled and phase10_auto_speak:
            print(f'[PHASE10 TTS INPUT] backend={phase10_voice.info.backend} full_sentence={sentence.text!r}', flush=True)
            queued=phase10_speaker.say(sentence.text)
            phase10_store.append('TTS_REQUEST',text=sentence.text,backend=phase10_voice.info.backend,queued=queued,status=phase10_speaker.last_status,reason='AUTO_FINALIZE')
        return sentence

    try:
        with holo.Holistic(static_image_mode=False,model_complexity=1,
                smooth_landmarks=True,refine_face_landmarks=True,
                min_detection_confidence=.5,min_tracking_confidence=.5) as detector:
            while True:
                ok,frame=cap.read()
                if not ok:
                    print('Camera read failed; stopping')
                    break
                now=time.perf_counter();dt=max(now-last_t,1e-6);last_t=now
                fps=(.9*fps+.1/dt) if fps else 1/dt
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                rgb.flags.writeable=False
                res=detector.process(rgb)
                sample=from_holistic(res,now)
                hand_present=bool(sample.left_present or sample.right_present)
                if hand_present: no_hand_frames=0
                else: no_hand_frames+=1
                release_pulse=release_scheduler.update(hand_present)
                if running:
                    stride_counter+=1
                    if gate.auto_release_no_hand and no_hand_frames>=release_scheduler.threshold_frames:
                        # Not a class prediction. A conservative heuristically detected
                        # release, logged as NO_HAND_RELEASE; never emit token itself.
                        buffer.clear()
                        if release_pulse:
                            submit('REST',1.0,now,'NO_HAND_RELEASE_HEURISTIC',None)
                    else:
                        buffer.append(sample)
                        if len(buffer)==win_frames and stride_counter>=stride:
                            stride_counter=0
                            try:
                                X,mask,features=build_model_input(
                                    list(buffer),cfg4,cfg5,cfg6,mean,std,target,feature_mode="spatial513")
                                t0=time.perf_counter()
                                probs=model.predict(X[np.newaxis,:,:],verbose=0)[0]
                                infer_ms=(time.perf_counter()-t0)*1000
                                latest=decide_window(probs,labels,list(buffer),gate)
                                spatial_last=None
                                if spatial_profile is not None:
                                    candidate32, valid32=v4_candidate32(list(buffer))
                                    spatial_last=evaluate_spatial_gate(spatial_profile,latest.label,candidate32,valid32)
                                    if spatial_last.decision in ("REJECT","INSUFFICIENT"):
                                        latest=replace(latest,label="UNKNOWN",reason=spatial_last.reason)
                                submit(latest.label,latest.confidence,now,'TCN_V3_SPATIAL_GATE' if spatial_profile else 'TCN_V3_SPATIAL',latest,infer_ms)
                            except Exception as exc:
                                print('[INFERENCE ERROR]',type(exc).__name__,exc)
                                latest=None
                                spatial_last=None
                annotated=frame.copy()
                if use_landmarks:
                    annotated=draw_landmarks(annotated,res,holo,drawing)
                if mirror:
                    annotated=cv2.flip(annotated,1)
                flags={
                    'buffer':len(buffer),'window':win_frames,'stride':stride,
                    'left':sample.left_present,'right':sample.right_present,
                    'pose':sample.pose_present,'face':sample.face_present,
                    'no_hands':no_hand_frames,'auto_release':gate.auto_release_no_hand,
                    'running':running,
                }
                display=render(annotated,labels,latest,engine,fps,infer_ms,
                               flags,width,height,gate,(now-last_event_time<.75),spatial_last)
                if phase10_builder is not None:
                    status=phase10_speaker.last_status
                    status+= ' | auto ON' if phase10_auto_speak else ' | auto OFF'
                    display=draw_phase10_banner(display,phase10_builder,status,enabled=phase10_banner)
                cv2.imshow(win,display)
                key=cv2.waitKey(1)&0xFF
                if key in (ord('q'),ord('Q'),27):
                    if trial_id and test_condition!='UNLABELED': trial_marker('TRIAL_END',now)
                    break
                if key in (ord('f'),ord('F')):
                    fullscreen=not fullscreen
                    cv2.setWindowProperty(win,cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:cv2.resizeWindow(win,width,height)
                elif key in (ord('v'),ord('V')):
                    use_landmarks=not use_landmarks
                elif key in (ord('m'),ord('M')):
                    mirror=not mirror
                elif key in (ord('p'),ord('P')):
                    running=not running
                    release_scheduler.reset()
                    buffer.clear();stride_counter=0;latest=None;spatial_last=None
                elif key in (ord('c'),ord('C')):
                    engine.reset();release_scheduler.reset();buffer.clear();stride_counter=0;latest=None;spatial_last=None
                    if phase10_builder is not None:
                        phase10_store.append('ACTIVE_CLEAR',reason='PHASE09_C_KEY',removed_tokens=list(phase10_builder.active))
                        phase10_builder.clear_active()
                        phase10_store.export(phase10_builder)
                    print('[CLEAR] confirmed tokens and pending sentence cleared (finalized history retained)')
                elif key in (ord('r'),ord('R')):
                    buffer.clear();stride_counter=0;latest=None;spatial_last=None
                    print('[RESET] window buffer reset (tokens retained)')
                elif key in (ord('b'),ord('B')):
                    # Manual release is for controlled debug only.
                    for j in range(engine.cfg.release_consecutive):
                        submit('REST',1.0,now + j*0.0001,'MANUAL_REST',None)
                    buffer.clear();stride_counter=0;latest=None;spatial_last=None
                    print('[MANUAL] REST release applied; not a genuine model result')
                elif key in (ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6')):
                    mapping={'0':'UNLABELED','1':'correct','2':'wrong_position',
                             '3':'wrong_handshape','4':'wrong_orientation','5':'partial','6':'REST'}
                    if trial_id and test_condition!='UNLABELED': trial_marker('TRIAL_END',now)
                    test_condition=mapping[chr(key)];trial_id+=1
                    buffer.clear();stride_counter=0;release_scheduler.reset()
                    if a.trial_isolation and test_condition!='UNLABELED':
                        engine.reset()  # Explicitly isolates trials; continuous mode is unchanged.
                        if phase10_builder is not None:
                            phase10_store.append('ACTIVE_CLEAR',reason='TRIAL_ISOLATION',removed_tokens=list(phase10_builder.active))
                            phase10_builder.clear_active()
                    if test_condition!='UNLABELED':trial_marker('TRIAL_START',now)
                    print('[TRIAL MARKER]',trial_id,test_condition,'target=',a.target_label)
                elif key in (ord('a'),ord('A')):
                    from dataclasses import replace
                    gate=replace(gate,auto_release_no_hand=not gate.auto_release_no_hand)
                    release_scheduler.reset()
                    print('[AUTO RELEASE]',gate.auto_release_no_hand)
                elif phase10_builder is not None and key in (ord('.'),ord('?'),ord('!'),ord(',')):
                    mark=chr(key)
                    if phase10_builder.add_punctuation(mark):
                        phase10_store.append('PUNCTUATION_ADDED',mark=mark,preview=phase10_builder.preview)
                        phase10_store.export(phase10_builder)
                        print('[PHASE10 PUNCTUATION]',phase10_builder.preview)
                    else:
                        print('[PHASE10 PUNCTUATION] No pending sentence')
                elif phase10_builder is not None and key in (ord('n'),ord('N'),32):
                    finalize_phase10()
                elif phase10_builder is not None and key in (ord('z'),ord('Z')):
                    undone=phase10_builder.undo()
                    if undone:phase10_store.append('UNDO_TEXT_TOKEN',label=undone,preview=phase10_builder.preview)
                    print('[PHASE10 UNDO]',undone or '(nothing to undo)')
                elif phase10_builder is not None and key in (ord('x'),ord('X')):
                    phase10_store.append('ACTIVE_CLEAR',reason='USER_X_KEY',removed_tokens=list(phase10_builder.active))
                    phase10_builder.clear_active();phase10_store.export(phase10_builder)
                    print('[PHASE10 CLEAR] Pending sentence cleared; finalized history kept')
                elif phase10_builder is not None and key in (ord('t'),ord('T')):
                    utterance=phase10_builder.preview or phase10_builder.last_text()
                    print(f'[PHASE10 TTS INPUT] backend={phase10_voice.info.backend} full_sentence={utterance!r}', flush=True)
                    queued=phase10_speaker.say(utterance)
                    phase10_store.append('TTS_REQUEST',text=utterance,backend=phase10_voice.info.backend,queued=queued,status=phase10_speaker.last_status,reason='USER_T_KEY')
                    print('[PHASE10 TTS]',phase10_speaker.last_status)
                elif phase10_builder is not None and key in (ord('y'),ord('Y')):
                    phase10_auto_speak=not phase10_auto_speak
                    phase10_store.append('AUTO_SPEAK_TOGGLE',enabled=phase10_auto_speak)
                    print('[PHASE10 AUTO TTS]',phase10_auto_speak)
                elif phase10_builder is not None and key in (ord('e'),ord('E')):
                    phase10_store.export(phase10_builder)
                    print('[PHASE10 EXPORTED]',phase10_store.transcript_path)
                elif key in (ord('s'),ord('S')):
                    snaps.mkdir(parents=True,exist_ok=True)
                    p=snaps/f"phase09_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(p),display)
                    print('[SNAPSHOT]',p)
    finally:
        if phase10_builder is not None:
            try:
                finalize_phase10(speak_if_enabled=False)  # Do not lose unfinalized token text on Quit.
                phase10_store.export(phase10_builder)
                phase10_store.append('SESSION_END',sentence_count=len(phase10_builder.sentences))
                print('[PHASE10 TRANSCRIPT]',phase10_store.transcript_path)
                print('[PHASE10 EVENT LOG]',phase10_store.event_path)
            finally:
                phase10_speaker.close()
        cap.release();cv2.destroyAllWindows()
    print('\nPHASE09 SESSION FINISHED')
    print('TOKENS:', ' '.join(x.label for x in engine.events) or '(none)')
    print('PREDICTIONS CSV:',csv_path)
    print('TOKEN EVENTS JSONL:',token_path)
    print('NOTE: Real webcam evaluation required; static/UNKNOWN gestures may still be misclassified.')


if __name__=='__main__':
    main()
