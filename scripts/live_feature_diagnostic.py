"""Webcam Feature Diagnostic for Phase09 Hand Position / Handshape.

Run from project root:
 python scripts/live_feature_diagnostic.py --label xin_chao --subject S01

Keys 1:correct 2:wrong_position 3:wrong_handshape 4:wrong_orientation
     5:partial 6:correct_repeat G:compare V:overlay M:mirror F:fullscreen Q:quit

Only records diagnostic clips; does NOT update trained model, dataset or labels.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.feature_phase4 import load_phase4_config
from core.feature_phase5 import load_phase5_config
from core.feature_phase6 import load_phase6_config
from core.phase8_live import FrameSample, build_model_input
from core.feature_diagnostic_video import (
    VideoSaveError, save_source_video, write_metadata, append_trial_manifest, next_sample_path,
)
from core.spatial_handshape_diagnostic import (
    CONDITIONS, extract_diagnostic_features,save_trial,pairwise_report,
)

COLORS={'left':(244,113,38),'right':(0,149,250),'pose':(57,210,99),'face':(199,95,230)}
KEY_CONDITIONS={ord('1'):'correct',ord('2'):'wrong_position',
                ord('3'):'wrong_handshape',ord('4'):'wrong_orientation',
                ord('5'):'partial',ord('6'):'correct_repeat'}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Phase09 Webcam Spatial/Handshape Diagnostic')
    p.add_argument('--label',default='xin_chao',help='Known class being tested')
    p.add_argument('--subject',default='S01',help='Signer identifier, e.g. S01')
    p.add_argument('--session',default='SES01',help='Session identifier to group recording sessions')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--frames',type=int,default=45)
    p.add_argument('--countdown',type=float,default=1.5)
    p.add_argument('--cam-width',type=int,default=1280)
    p.add_argument('--cam-height',type=int,default=720)
    p.add_argument('--output',default='logs/feature_diagnostic')
    p.add_argument('--fullscreen',action='store_true')
    video=p.add_mutually_exclusive_group()
    video.add_argument('--save-video',dest='save_video',action='store_true',default=True,
                       help='Save source video (DEFAULT; backwards compatible)')
    video.add_argument('--no-video',dest='save_video',action='store_false',
                       help='Explicitly disable video (use for debugging only)')
    p.add_argument('--video-format',choices=('avi','mp4'),default='avi',
                   help='AVI MJPG (default, reliable local playback) or MP4 mp4v with AVI fallback')
    p.add_argument('--with-model',action='store_true',help='Optional frozen TCN V2 prediction')
    p.add_argument('--model-path',default='artifacts/experiments/tcn_v2_unknown/model.keras')
    p.add_argument('--stats-path',default='artifacts/experiments/tcn_v2_unknown/preprocess_stats.npz')
    p.add_argument('--labels-path',default='artifacts/experiments/tcn_v2_unknown/labels.json')
    p.add_argument('--dry-run',action='store_true',help='Check configs/dependencies without opening camera')
    return p.parse_args(argv)


def rp(x):
    p=Path(x);return p if p.is_absolute() else ROOT/p


def from_mediapipe(result, timestamp):
    def coords(obj,n):
        if obj is None or len(obj.landmark)<n:
            return np.zeros((n,3),np.float32),False
        arr=np.asarray([(p.x,p.y,p.z) for p in obj.landmark[:n]],np.float32)
        if arr.shape!=(n,3) or not np.isfinite(arr).all():
            return np.zeros((n,3),np.float32),False
        return arr,True
    left,lp=coords(result.left_hand_landmarks,21)
    right,rp_=coords(result.right_hand_landmarks,21)
    pose,pp=coords(result.pose_landmarks,33)
    face,fp=coords(result.face_landmarks,468)
    return FrameSample(left,right,pose,face,lp,rp_,pp,fp,float(timestamp))


def add_overlay(im,result,holistic,drawing):
    def spec(color,r=2,th=2):
        return drawing.DrawingSpec(color=color,thickness=th,circle_radius=r)
    if result.pose_landmarks:
        drawing.draw_landmarks(im,result.pose_landmarks,holistic.POSE_CONNECTIONS,
                               spec(COLORS['pose'],2),spec(COLORS['pose'],1,1))
    if result.face_landmarks:
        drawing.draw_landmarks(im,result.face_landmarks,holistic.FACEMESH_CONTOURS,
                               None,spec(COLORS['face'],1,1))
    if result.left_hand_landmarks:
        drawing.draw_landmarks(im,result.left_hand_landmarks,holistic.HAND_CONNECTIONS,
                               spec(COLORS['left'],4,3),spec(COLORS['left'],2))
    if result.right_hand_landmarks:
        drawing.draw_landmarks(im,result.right_hand_landmarks,holistic.HAND_CONNECTIONS,
                               spec(COLORS['right'],4,3),spec(COLORS['right'],2))
    return im


def annotate_metrics(im, sample):
    if not sample.pose_present:return im
    h,w=im.shape[:2]
    p=sample.pose
    center=(p[11,:2]+p[12,:2])*0.5
    pc=(int(center[0]*w),int(center[1]*h))
    cv2.circle(im,pc,7,(220,215,20),-1)
    for hand,present,color in ((sample.left,sample.left_present,COLORS['left']),
                               (sample.right,sample.right_present,COLORS['right'])):
        if present:
            wrist=(int(hand[0,0]*w),int(hand[0,1]*h))
            cv2.circle(im,wrist,8,color,-1)
            cv2.line(im,wrist,pc,color,2,cv2.LINE_AA)
    return im


def fit_contain(frame,width,height):
    h,w=frame.shape[:2]
    sf=min(width/max(w,1),height/max(h,1))
    nw=max(1,int(round(w*sf)));nh=max(1,int(round(h*sf)))
    scaled=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA if sf<1 else cv2.INTER_LINEAR)
    canvas=np.full((height,width,3),(16,21,29),np.uint8)
    x=(width-nw)//2;y=(height-nh)//2
    canvas[y:y+nh,x:x+nw]=scaled
    return canvas


def draw(frame,sample,condition,count,frames,seconds_left,live_status,
         fps,last_info,mirrored,fullscreen):
    width,height=1500,850
    cam_width=1035
    canvas=np.full((height,width,3),(29,36,48),np.uint8)
    canvas[:,:cam_width]=fit_contain(frame,cam_width,height)
    cv2.line(canvas,(cam_width,0),(cam_width,height),(105,118,136),2)
    x=cam_width+17;y=42
    def text(msg,color=(230,238,248),size=.58,dy=31):
        nonlocal y
        cv2.putText(canvas,str(msg),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,2 if size>.65 else 1,cv2.LINE_AA)
        y+=dy
    text('PHASE09 FEATURE DIAGNOSTIC',(110,230,244),.7,42)
    text(f'FPS: {fps:.1f}',(180,200,223),.53)
    text('LANDMARK STATUS',size=.61,dy=29)
    text(f"L: {'ON' if sample.left_present else 'OFF'} | R: {'ON' if sample.right_present else 'OFF'}",size=.5)
    text(f"P: {'ON' if sample.pose_present else 'OFF'} | F: {'ON' if sample.face_present else 'OFF'}",size=.5,dy=40)
    text('CAPTURE CONDITION',size=.61,dy=27)
    text(condition or '(Press 1-6)',(25,198,255),.6,dy=33)
    text(live_status,(100,225,160) if count==frames else (20,206,253),.62,dy=35)
    text(f'Buffer: {count}/{frames}',size=.56,dy=30)
    if seconds_left>0:text(f'Countdown: {seconds_left:.1f}s',(12,225,247),.61,dy=33)
    y+=5
    text('1 Correct',size=.48,dy=25)
    text('2 Wrong position',size=.48,dy=25)
    text('3 Wrong handshape',size=.48,dy=25)
    text('4 Wrong orientation',size=.48,dy=25)
    text('5 Partial gesture',size=.48,dy=25)
    text('6 Correct repeat',size=.48,dy=32)
    text('G Compare saved clips',size=.48,dy=26)
    text('V Overlay | M Mirror',size=.47,dy=26)
    text('F Fullscreen | Q Quit',size=.47,dy=36)
    if last_info:
        text('LAST SAVED',size=.52,dy=27)
        text(last_info.get('condition','')[:24],(100,236,180),.48,dy=25)
        text('TCN: '+last_info.get('model_prediction','(not enabled)')[:22],size=.48,dy=23)
        text('QC any_hand: '+f"{last_info.get('any_hand_ratio',0):.2f}",size=.48,dy=25)
    cv2.rectangle(canvas,(0,height-45),(cam_width,height),(15,22,31),-1)
    cv2.putText(canvas,'BLUE LEFT | ORANGE RIGHT | GREEN POSE | PURPLE FACE',(18,height-17),
                cv2.FONT_HERSHEY_SIMPLEX,.62,(235,237,244),2,cv2.LINE_AA)
    return canvas


def load_optional_model(a):
    if not a.with_model:return None
    model_path=rp(a.model_path);stats_path=rp(a.stats_path);labels_path=rp(a.labels_path)
    for p in (model_path,stats_path,labels_path):
        if not p.is_file():raise FileNotFoundError(f'Missing optional model artifact: {p}')
    labels_obj=json.loads(labels_path.read_text(encoding='utf-8'))
    labels=labels_obj['labels'] if isinstance(labels_obj,dict) else labels_obj
    if not isinstance(labels,list) or len(set(labels))!=len(labels) or len(labels)!=5:
        raise ValueError('TCN V2 labels should be a 5-element list')
    with np.load(stats_path,allow_pickle=False) as d:
        mean=np.asarray(d['mean'],np.float32)
        std=np.asarray(d['std'],np.float32)
        target=int(np.asarray(d['target_frames']).reshape(-1)[0])
        dim=int(np.asarray(d['feature_dim']).reshape(-1)[0])
    if target!=32 or dim!=453 or mean.shape!=(453,) or std.shape!=(453,) or np.any(std<=0):
        raise ValueError('Frozen scaler incompatible with 32x453')
    import tensorflow as tf
    model=tf.keras.models.load_model(model_path,compile=False)
    ishape=model.input_shape[0] if isinstance(model.input_shape,list) else model.input_shape
    if tuple(ishape[1:])!=(32,453) or model.output_shape[-1]!=5:
        raise ValueError(f'Unexpected model shape {ishape} / {model.output_shape}')
    return {'model':model,'labels':labels,'mean':mean,'std':std,'target':target,'path':str(model_path)}


def main():
    a=parse_args()
    if a.frames<8:raise ValueError('--frames must be >=8')
    if a.countdown<0:raise ValueError('--countdown >=0 required')
    if not a.label.replace('_','').replace('-','').isalnum():
        raise ValueError('--label accepts letters/numbers/underscore/hyphen only')
    if not a.subject.replace('_','').replace('-','').isalnum():
        raise ValueError('--subject accepts letters/numbers/underscore/hyphen only')
    if not a.session.replace('_','').replace('-','').isalnum():
        raise ValueError('--session accepts letters/numbers/underscore/hyphen only')
    cfg4,_=load_phase4_config(rp('configs/feature_phase4.yaml'))
    cfg5,_=load_phase5_config(rp('configs/feature_phase5.yaml'))
    cfg6,_=load_phase6_config(rp('configs/feature_phase6.yaml'))
    model=load_optional_model(a)
    if a.dry_run:
        print('DIAGNOSTIC SETUP: PASS (configs available, model optional)')
        return
    try:
        import mediapipe as mp
        holistic=mp.solutions.holistic
        drawing=mp.solutions.drawing_utils
    except (ImportError,AttributeError) as exc:
        raise RuntimeError('Install mediapipe==0.10.21 in compatible venv; mp.solutions required') from exc
    cap=cv2.VideoCapture(a.camera,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(a.camera)
    if not cap.isOpened():cap=cv2.VideoCapture(a.camera)
    if not cap.isOpened():raise RuntimeError('Cannot open webcam; try --camera 1')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.cam_height)
    cap.set(cv2.CAP_PROP_FPS,30)
    window='KLTN | Phase09 Spatial Handshape Diagnostic'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1500,850)
    fullscreen=a.fullscreen
    if fullscreen:cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    overlay=True;mirror=True
    condition=None;deadline=0;capturing=False;record=[];vid_frames=[]
    status='READY';last_info={};last=time.perf_counter();fps=0.
    out=rp(a.output)
    try:
        with holistic.Holistic(static_image_mode=False,model_complexity=1,
                smooth_landmarks=True, refine_face_landmarks=True,
                min_detection_confidence=.5,min_tracking_confidence=.5) as detector:
            while True:
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Webcam frame read failed')
                now=time.perf_counter();dt=max(now-last,1e-6);last=now
                fps=0.85*fps+0.15/dt if fps else 1/dt
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                rgb.flags.writeable=False
                results=detector.process(rgb)
                sample=from_mediapipe(results,now)
                remain=max(0,deadline-now) if condition and not capturing else 0
                if condition and not capturing and now>=deadline:
                    capturing=True;status='RECORDING';record=[];vid_frames=[]
                if capturing:
                    record.append(sample)
                    if a.save_video:vid_frames.append(frame.copy())
                    if len(record)>=a.frames:
                        status='PROCESSING'
                        try:
                            raw,features,qc=extract_diagnostic_features(record,cfg4,cfg5,cfg6)
                            dest, sample_id = next_sample_path(out,a.label,a.subject,a.session,condition)
                            meta={
                                'sample_id':sample_id,
                                'captured_at':datetime.now(timezone.utc).isoformat(),
                                'label':a.label,'subject':a.subject,
                                'subject_id':a.subject,
                                'session':a.session,'session_id':a.session,
                                'condition':condition,'frames':len(record),
                                'hand_tracking_convention':'unmirrored frames, MediaPipe Holistic labels',
                                'feature_pipeline':'Phase04-Phase05-Phase06 unchanged + 32 candidate-only diagnostic dims',
                                'qc':qc,
                            }
                            if model:
                                X,m,old=build_model_input(record,cfg4,cfg5,cfg6,model['mean'],model['std'],model['target'])
                                if not np.allclose(old,features['features453'],atol=1e-5):
                                    raise RuntimeError('Diagnostic/model feature inconsistency')
                                prob=np.asarray(model['model'].predict(X[np.newaxis],verbose=0)[0],np.float64)
                                idx=int(np.argmax(prob))
                                meta.update(model_prediction=model['labels'][idx],model_confidence=float(prob[idx]),
                                            model_probabilities={k:float(v) for k,v in zip(model['labels'],prob)},
                                            model_path=model['path'])
                            # Landmark/features and source video share the *same* trial ID.
                            # Even if encoding fails, retain the raw sample and mark it
                            # VIDEO_FAILED (never silently treat a missing video as saved).
                            save_trial(dest,raw,features,meta)
                            if a.save_video:
                                try:
                                    meta['video']=save_source_video(
                                        dest,vid_frames,[f.timestamp for f in record],
                                        preferred=a.video_format)
                                except (VideoSaveError,OSError,ValueError) as error:
                                    meta['video']={'status':'FAILED','error':str(error),
                                                   'filename':None}
                            else:
                                meta['video']={'status':'DISABLED','filename':None}
                            write_metadata(dest,meta)
                            append_trial_manifest(out,dest,meta)
                            video_status=meta['video']['status']
                            if video_status=='FAILED':
                                status='VIDEO FAILED - SEE TERMINAL'
                                print('[VIDEO FAILED]',dest,meta['video']['error'])
                            else:
                                status='SAVED + VIDEO' if a.save_video else 'SAVED (NO VIDEO)'
                                if a.save_video:
                                    print('[VIDEO VERIFIED]', dest/meta['video']['filename'],
                                          'frames',meta['video']['decoded_frames'])
                            last_info={'condition':condition,**qc,**{k:meta[k] for k in ('model_prediction','model_confidence') if k in meta}}
                            print('[SAVED]',dest, 'HAND',f"{qc['any_hand_ratio']:.2f}",
                                  'TCN',meta.get('model_prediction','disabled'),
                                  meta.get('model_confidence',''))
                        except Exception as exc:
                            print('[CAPTURE FAILED]',type(exc).__name__,exc)
                            status='ERROR - SEE TERMINAL'
                        condition=None;capturing=False;record=[];vid_frames=[]
                annotated=frame.copy()
                if overlay:
                    annotated=add_overlay(annotated,results,holistic,drawing)
                    annotated=annotate_metrics(annotated,sample)
                if mirror:annotated=cv2.flip(annotated,1)
                screen=draw(annotated,sample,condition,len(record),a.frames,remain,status,fps,last_info,mirror,fullscreen)
                # Identity is displayed in preview only; saved source video is clean/unmirrored.
                cv2.rectangle(screen,(8,8),(1014,43),(24,30,40),-1)
                identity=f'SIGN {a.label}  |  SIGNER {a.subject}  |  SESSION {a.session}  |  VIDEO {"ON" if a.save_video else "OFF"}'
                cv2.putText(screen,identity[:112],(17,32),cv2.FONT_HERSHEY_SIMPLEX,.61,(250,250,250),2,cv2.LINE_AA)
                cv2.imshow(window,screen)
                key=cv2.waitKey(1)&0xff
                if key in (ord('q'),ord('Q'),27):break
                if key in KEY_CONDITIONS:
                    condition=KEY_CONDITIONS[key];deadline=time.perf_counter()+a.countdown
                    capturing=False;record=[];vid_frames=[];status='COUNTDOWN'
                    print('[PREPARE]',condition,'label',a.label,'subject',a.subject)
                elif key in (ord('g'),ord('G')):
                    try:
                        rows,path=pairwise_report(out,a.label,a.subject,out/'reports')
                        print('[COMPARE]',path,'groups:',len(rows))
                        for row in rows:
                            if row['condition'] in ('wrong_position','wrong_handshape') and row['group'] in (
                                    'hand_position126','finger_angles30','hand_body_position12','finger_curl10'):
                                print(row['condition'],row['group'],'RMS',row['rms'],'valid',row['compared_ratio'])
                        status='COMPARE SAVED'
                    except Exception as exc:
                        print('[COMPARE ERROR]',type(exc).__name__,exc)
                        status='NEED CORRECT + VARIANT'
                elif key in (ord('v'),ord('V')):overlay=not overlay
                elif key in (ord('m'),ord('M')):mirror=not mirror
                elif key in (ord('f'),ord('F')):
                    fullscreen=not fullscreen
                    cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:cv2.resizeWindow(window,1500,850)
    finally:
        cap.release();cv2.destroyAllWindows()
    print('Output:',out)
    print('Video recording:', 'ON (default)' if a.save_video else 'OFF (--no-video)')
    print('Manifest:',out/'manifest.csv')
    print('Run comparison anytime with: python scripts/compare_feature_diagnostic.py --label',a.label,'--subject',a.subject)

if __name__=='__main__':main()
