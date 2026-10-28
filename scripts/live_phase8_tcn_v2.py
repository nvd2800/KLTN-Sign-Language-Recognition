"""Phase 08 live KNOWN/UNKNOWN tester — camera/landmark/TCN/failure logging.

Run from project root:
    python scripts/live_phase8_camera.py --fullscreen

Key 0: capture UNKNOWN trial; keys 1..9: capture a known-class trial.
SPACE: capture unlabeled trial. P: toggle rolling PREVIEW. Q: quit.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from core.feature_phase4 import load_phase4_config
from core.feature_phase5 import load_phase5_config
from core.feature_phase6 import load_phase6_config
from core.phase8_live import (
    FrameSample, append_csv_row, build_model_input, judge_trial,
    load_threshold, phase08_decision,
)


def at_root(s):
    p=Path(s)
    return p if p.is_absolute() else ROOT/p


def parse_args():
    ap=argparse.ArgumentParser(description="Phase08 realtime camera KNOWN vs UNKNOWN tester")
    ap.add_argument("--config",default="configs/phase8_unknown.yaml")
    ap.add_argument("--camera",type=int,default=0)
    ap.add_argument("--model",default="artifacts/experiments/tcn_v2_unknown/model.keras",help="TCN V2 model path")
    ap.add_argument("--stats",default="artifacts/experiments/tcn_v2_unknown/preprocess_stats.npz",help="V2 preprocessing stats")
    ap.add_argument("--labels",default="artifacts/experiments/tcn_v2_unknown/labels.json",help="Five-class label map")
    ap.add_argument("--threshold-file",default="artifacts/phase8_unknown/calibrated_threshold.json")
    ap.add_argument("--threshold",type=float,default=0.0,help="Use direct TCN V2 five-class softmax (default 0.0)")
    ap.add_argument("--min-margin",type=float,default=0.0,help="Experimental margin gate, default OFF")
    ap.add_argument("--frames",type=int,default=45,help="Capture/rolling sequence length")
    ap.add_argument("--preview-stride",type=int,default=8,help="Infer every N new frames in preview")
    ap.add_argument("--countdown",type=float,default=1.0,help="Seconds from key press to capture start")
    ap.add_argument("--fullscreen",action="store_true")
    ap.add_argument("--width",type=int,default=1480)
    ap.add_argument("--height",type=int,default=860)
    ap.add_argument("--cam-width",type=int,default=1280)
    ap.add_argument("--cam-height",type=int,default=720)
    ap.add_argument("--cam-fps",type=int,default=30)
    ap.add_argument("--no-preview",action="store_true",help="Disable rolling predictions")
    ap.add_argument("--no-mirror",action="store_true",help="Don't mirror visual preview")
    ap.add_argument("--no-save-trials",action="store_true",help="Only log CSV, no raw landmark NPZ")
    ap.add_argument("--log",default="logs/phase08_live_camera.csv")
    ap.add_argument("--trial-dir",default="logs/phase08_live_trials")
    ap.add_argument("--snapshot-dir",default="logs/phase08_live_snapshots")
    return ap.parse_args()


def pick_artifacts(cfg,args):
    experiment=at_root(cfg["model"]["experiment_dir"])
    model_candidates=([at_root(args.model)] if args.model else [
        experiment/"model.keras",at_root(cfg["model"]["fallback_model"]),
    ])
    model_path=next((p for p in model_candidates if p.is_file()),None)
    if model_path is None:
        raise FileNotFoundError(f"TCN model not found. Checked: {model_candidates}")
    preferred=[experiment/"preprocess_stats.npz",at_root(cfg["model"]["preprocess_stats"])]
    stats_path=at_root(args.stats) if args.stats else next((p for p in preferred if p.is_file()),None)
    label_candidates=[experiment/"labels.json",at_root(cfg["model"]["labels_json"])]
    labels_path=at_root(args.labels) if args.labels else next((p for p in label_candidates if p.is_file()),None)
    if stats_path is None or not stats_path.is_file():
        raise FileNotFoundError("Missing frozen scaler preprocess_stats.npz; set --stats")
    if labels_path is None or not labels_path.is_file():
        raise FileNotFoundError("Missing labels.json; set --labels")
    # Avoid mixing frozen model with newer scaler/labels after V2 training.
    if model_path == experiment/"model.keras" and not args.stats and stats_path != preferred[0]:
        raise RuntimeError("Frozen TCN V1 found but frozen scaler missing. Run freeze_tcn_v1.py or provide --stats")
    if model_path == experiment/"model.keras" and not args.labels and labels_path != label_candidates[0]:
        raise RuntimeError("Frozen TCN V1 found but frozen labels missing. Run freeze_tcn_v1.py or provide --labels")
    return model_path,stats_path,labels_path


def as_xyz(landmarks, points):
    if landmarks is None or len(landmarks.landmark) < points:
        return np.zeros((points,3),np.float32),False
    xyz=np.asarray([(p.x,p.y,p.z) for p in landmarks.landmark[:points]],np.float32)
    if xyz.shape!=(points,3) or not np.isfinite(xyz).all():
        return np.zeros((points,3),np.float32),False
    return xyz,True


def frame_from_holistic(res, timestamp):
    l,lp=as_xyz(res.left_hand_landmarks,21)
    r,rp=as_xyz(res.right_hand_landmarks,21)
    p,pp=as_xyz(res.pose_landmarks,33)
    f,fp=as_xyz(res.face_landmarks,468)
    return FrameSample(l,r,p,f,lp,rp,pp,fp,float(timestamp))


def box_text(im,txt,x,y,color=(245,245,245),scale=0.55,thickness=1):
    cv2.putText(im,str(txt),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def image_fit(src,W,H):
    """Fit without cropping, letterbox where needed."""
    h,w=src.shape[:2]
    factor=min(W/max(1,w),H/max(1,h))
    tw=max(1,int(w*factor)); th=max(1,int(h*factor))
    interp=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR
    img=cv2.resize(src,(tw,th),interpolation=interp)
    target=np.full((H,W,3),(10,13,22),np.uint8)
    x=(W-tw)//2;y=(H-th)//2
    target[y:y+th,x:x+tw]=img
    return target


def draw_landmarks(frame,res,mp,draw,draw_on=True):
    if not draw_on:
        return frame
    blue=draw.DrawingSpec(color=(255,100,0),thickness=3,circle_radius=3)
    orange=draw.DrawingSpec(color=(0,160,255),thickness=3,circle_radius=3)
    green=draw.DrawingSpec(color=(25,210,65),thickness=2,circle_radius=2)
    violet=draw.DrawingSpec(color=(190,70,235),thickness=1,circle_radius=1)
    conn_blue=draw.DrawingSpec(color=(255,180,80),thickness=2,circle_radius=1)
    conn_orange=draw.DrawingSpec(color=(65,210,255),thickness=2,circle_radius=1)
    if res.pose_landmarks is not None:
        draw.draw_landmarks(frame,res.pose_landmarks,mp.POSE_CONNECTIONS,green,green)
    if res.face_landmarks is not None:
        draw.draw_landmarks(frame,res.face_landmarks,mp.FACEMESH_CONTOURS,None,violet)
    if res.left_hand_landmarks is not None:
        draw.draw_landmarks(frame,res.left_hand_landmarks,mp.HAND_CONNECTIONS,blue,conn_blue)
    if res.right_hand_landmarks is not None:
        draw.draw_landmarks(frame,res.right_hand_landmarks,mp.HAND_CONNECTIONS,orange,conn_orange)
    return frame


def render_view(annotated, uiw, uih, labels, status, latest_preview, latest_trial,
                threshold, threshold_source, confidence_warning, fps, camera_presence,
                capture_gt, remaining, preview_enabled, model_ms, can_predict):
    pw=max(370,int(uiw*.28)); vw=uiw-pw
    base=np.full((uih,uiw,3),(25,28,36),np.uint8)
    view=image_fit(annotated,vw,uih)
    base[:,:vw]=view
    cv2.rectangle(base,(vw,0),(uiw,uih),(31,34,45),-1)
    x=vw+18;y=33
    box_text(base,"PHASE 08 | CAMERA TEST",x,y,(255,255,255),.62,2); y+=28
    box_text(base,f"TCN V1  |  {fps:.1f} camera FPS",x,y,(205,209,222),.49);y+=30
    box_text(base,f"TCN V2 5 classes | gate: {threshold:.3f}",x,y,(100,230,230),.48);y+=25
    if confidence_warning:
        box_text(base,"! OOV VAL rejection=0%",x,y,(70,70,255),.48,2);y+=22
        box_text(base,"  Threshold NOT reliable",x,y,(70,70,255),.46,2);y+=22
    elif threshold_source == "manual override":
        box_text(base,"! MANUAL threshold (experiment)",x,y,(30,180,255),.45,2);y+=25
    else:
        y+=8
    box_text(base,f"L-hand: {'ON' if camera_presence[0] else 'OFF'}",x,y,(255,160,25) if camera_presence[0] else (90,105,135),.56,2);y+=25
    box_text(base,f"R-hand: {'ON' if camera_presence[1] else 'OFF'}",x,y,(0,175,255) if camera_presence[1] else (90,105,135),.56,2);y+=25
    box_text(base,f"Pose: {'ON' if camera_presence[2] else 'OFF'}",x,y,(45,220,90) if camera_presence[2] else (90,105,135),.55,2);y+=25
    box_text(base,f"Face: {'ON' if camera_presence[3] else 'OFF'}",x,y,(195,90,235) if camera_presence[3] else (90,105,135),.55,2);y+=34
    cv2.line(base,(x,y),(uiw-12,y),(100,105,115),1);y+=24

    if capture_gt is not None:
        box_text(base,f"CAPTURING GT: {capture_gt or 'unlabeled'}",x,y,(0,215,255),.57,2); y+=23
        box_text(base,f"Frames remaining: {remaining}",x,y,(0,215,255),.56);y+=22
    else:
        box_text(base,"MODE: ROLLING PREVIEW" if preview_enabled else "MODE: CAPTURE ONLY",x,y,(210,215,230),.53,2);y+=27

    def result_panel(d,heading,top_y):
        if d is None:
            box_text(base,f"{heading}: waiting",x,top_y,(150,160,170),.48)
            return top_y+27
        box_text(base,heading,x,top_y,(225,225,235),.51,2);top_y+=27
        cc=(0,225,85) if d.outcome!="UNKNOWN" else (60,170,255)
        box_text(base,f"OUT: {d.outcome}",x,top_y,cc,.76,2);top_y+=30
        box_text(base,f"RAW: {d.top1}  {d.confidence*100:.2f}%",x,top_y,(210,210,210),.50);top_y+=22
        box_text(base,f"Margin: {d.margin:.4f}  |  {model_ms:.0f} ms",x,top_y,(195,195,195),.46);top_y+=28
        if heading=="PREVIEW (NOT CONFIRMED)":
            for name,p in d.probabilities.items():
                box_text(base,name[:16],x,top_y,(223,223,223),.46)
                bar_x=x+130;bar_w=max(65,uiw-bar_x-70)
                cv2.rectangle(base,(bar_x,top_y-12),(bar_x+bar_w,top_y),(70,73,83),-1)
                cv2.rectangle(base,(bar_x,top_y-12),(bar_x+int(bar_w*min(1,max(0,p))),top_y),(66,195,218),-1)
                box_text(base,f"{p*100:4.1f}%",bar_x+bar_w+5,top_y,(226,226,226),.36)
                top_y+=23
        return top_y
    y=result_panel(latest_preview,"PREVIEW (NOT CONFIRMED)",y)
    y+=8
    if latest_trial is not None:
        d,gt,verdict=latest_trial
        y=result_panel(d,"LAST CAPTURE RESULT",y)
        box_text(base,f"GT: {gt or 'unlabeled'} | {verdict}",x,y,
                 (50,215,80) if verdict=="CORRECT" else (80,100,255),.43,2)
        y+=28
    footer=uih-125
    cv2.rectangle(base,(vw,footer),(uiw,uih),(22,25,34),-1)
    guides=["0: UNKNOWN trial | SPACE: unlabeled",
            "1..9: label trial (see console)",
            "P: rolling ON/OFF | V: landmarks",
            "F: fullscreen | R: reset | S: snapshot",
            "Q/ESC: exit"]
    for k,line in enumerate(guides):
        box_text(base,line,x,footer+19+20*k,(220,220,225),.43)
    if not can_predict:
        box_text(base,"Collecting frames ...",17,uih-26,(0,240,245),.65,2)
    return base


def write_trial_npz(path, samples, X, mask, raw453):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,
        timestamps=np.asarray([z.timestamp for z in samples],np.float64),
        left_hand=np.stack([z.left for z in samples]),
        right_hand=np.stack([z.right for z in samples]),
        pose=np.stack([z.pose for z in samples]),
        face=np.stack([z.face for z in samples]),
        left_present=np.array([z.left_present for z in samples],bool),
        right_present=np.array([z.right_present for z in samples],bool),
        pose_present=np.array([z.pose_present for z in samples],bool),
        face_present=np.array([z.face_present for z in samples],bool),
        features453=raw453,
        temporal32=X,
        temporal_mask=mask,
    )


def main():
    args=parse_args()
    if args.frames < 32 or args.preview_stride < 1 or args.countdown < 0:
        raise SystemExit("--frames >= 32, --preview-stride >= 1 and --countdown >= 0 required")
    cfg=yaml.safe_load(at_root(args.config).read_text(encoding="utf-8"))
    model_path,stats_path,labels_path=pick_artifacts(cfg,args)
    labels_obj=json.loads(labels_path.read_text(encoding="utf-8"))
    labels=labels_obj["labels"] if isinstance(labels_obj,dict) else labels_obj
    if not isinstance(labels,list) or len(labels)!=5 or "UNKNOWN" not in labels:
        raise RuntimeError("TCN V2 requires exactly five labels including UNKNOWN")
    if len(labels)>9:
        raise RuntimeError("Keyboard capture supports up to nine known classes")
    with np.load(stats_path,allow_pickle=False) as d:
        mean=np.asarray(d["mean"],np.float32)
        std=np.asarray(d["std"],np.float32)
        target_frames=int(np.asarray(d["target_frames"]).reshape(-1)[0])
        feature_dim=int(np.asarray(d["feature_dim"]).reshape(-1)[0])
    if target_frames!=32 or feature_dim!=453:
        raise RuntimeError(f"Expected TCN 32x453; scaler is {target_frames}x{feature_dim}")
    cfg4,_=load_phase4_config(at_root("configs/feature_phase4.yaml"))
    cfg5,_=load_phase5_config(at_root("configs/feature_phase5.yaml"))
    cfg6,_=load_phase6_config(at_root("configs/feature_phase6.yaml"))
    threshold,threshold_source,validation=load_threshold(at_root(args.threshold_file),args.threshold)
    confidence_warning=(validation is not None and
        float(validation.get("unknown_reject_rate",1)) <= 0.0)

    print("\nPHASE08 LIVE CAMERA TEST")
    print("Model:",model_path)
    print("Scaler:",stats_path)
    print("Labels:",labels)
    print("Threshold:",threshold,"source:",threshold_source)
    if confidence_warning:
        print("WARNING: UNKNOWN VAL rejection=0%; this threshold is NOT reliable for OOV.")
    print("Keys: 0=UNKNOWN; SPACE=unlabeled; known GT:",
          ", ".join(f"{i+1}={s}" for i,s in enumerate(labels) if s != "UNKNOWN"))
    print("      P=rolling preview, V=toggle landmarks, F=fullscreen, R=reset, S=screenshot, Q=exit")

    try:
        import mediapipe as mp
        holistic_api=mp.solutions.holistic
        draw=mp.solutions.drawing_utils
    except (ImportError,AttributeError) as exc:
        raise RuntimeError("MediaPipe 'solutions.holistic' missing. Check mediapipe/protobuf env and python -m pip check") from exc

    print("Loading model, please wait ...")
    import tensorflow as tf
    model=tf.keras.models.load_model(model_path,compile=False)
    shape=model.input_shape
    if isinstance(shape,list):
        shape=shape[0]
    if len(shape)!=3 or tuple(shape[1:])!=(32,453):
        raise RuntimeError(f"Model input_shape={shape}, expected (None,32,453)")
    if model.output_shape[-1]!=len(labels):
        raise RuntimeError(f"Model output classes={model.output_shape[-1]} != labels={len(labels)}")
    print("Model ready!")

    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW) if sys.platform=="win32" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Webcam unavailable. Try --camera 1 and close other camera apps")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.cam_height)
    cap.set(cv2.CAP_PROP_FPS,args.cam_fps)
    win="Phase 08 | Realtime KNOWN vs UNKNOWN (TCN V2 trained)"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win,args.width,args.height)
    fullscreen=args.fullscreen
    if fullscreen:
        cv2.setWindowProperty(win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    preview_enabled=not args.no_preview
    draw_enabled=True
    mirror=not args.no_mirror
    rolling=deque(maxlen=args.frames)
    new_frames=0
    capture=None
    capture_gt=None
    capture_start=0.0
    latest_preview=None
    latest_trial=None
    latest_ms=0.0
    fps=0.0; last_time=time.perf_counter()
    trial_index=0
    log_path=at_root(args.log)
    savedir=at_root(args.trial_dir)
    snapdir=at_root(args.snapshot_dir)

    def predict(samples):
        nonlocal latest_ms
        X,mask,raw453=build_model_input(samples,cfg4,cfg5,cfg6,mean,std,target_frames)
        st=time.perf_counter()
        prob=model.predict(X[np.newaxis,:,:],verbose=0)
        latest_ms=(time.perf_counter()-st)*1000
        d=phase08_decision(np.asarray(prob)[0],labels,threshold,args.min_margin)
        return d,X,mask,raw453

    try:
        with holistic_api.Holistic(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            refine_face_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as h:
            while True:
                ok,frame=cap.read()
                if not ok:
                    print("WARNING: camera.read failed; exiting")
                    break
                now=time.perf_counter()
                step=max(now-last_time,1e-6)
                fps=(0.88*fps+0.12*(1/step)) if fps>0 else 1/step
                last_time=now
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                rgb.flags.writeable=False
                res=h.process(rgb)
                sample=frame_from_holistic(res,now)
                rolling.append(sample);new_frames+=1
                if capture is not None and now>=capture_start:
                    capture.append(sample)
                    if len(capture)>=args.frames:
                        try:
                            decision,X,mask,raw453=predict(capture)
                            verdict=judge_trial(capture_gt,decision)
                            latest_trial=(decision,capture_gt,verdict)
                            trial_index+=1
                            stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                            trial_id=f"phase08_{stamp}_{trial_index:03d}"
                            n_l=sum(s.left_present for s in capture)/len(capture)
                            n_r=sum(s.right_present for s in capture)/len(capture)
                            n_p=sum(s.pose_present for s in capture)/len(capture)
                            n_f=sum(s.face_present for s in capture)/len(capture)
                            row={
                                "trial_id":trial_id,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
                                "gt":capture_gt or "","raw_top1":decision.top1,
                                "result":decision.outcome,"confidence":f"{decision.confidence:.6f}",
                                "margin":f"{decision.margin:.6f}","threshold":f"{threshold:.6f}",
                                "min_margin":f"{args.min_margin:.6f}","verdict":verdict,
                                "frames":len(capture),"inference_ms":f"{latest_ms:.3f}",
                                "left_presence_ratio":f"{n_l:.4f}","right_presence_ratio":f"{n_r:.4f}",
                                "pose_presence_ratio":f"{n_p:.4f}","face_presence_ratio":f"{n_f:.4f}",
                                "model_path":str(model_path),
                            }
                            for name,p in decision.probabilities.items():
                                row["p_"+name]=f"{p:.6f}"
                            append_csv_row(log_path,row,list(row))
                            if not args.no_save_trials:
                                write_trial_npz(savedir/(trial_id+".npz"),capture,X,mask,raw453)
                                (savedir/(trial_id+".json")).write_text(
                                    json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
                            print(f"[TRIAL {trial_index}] GT={capture_gt or 'unlabeled':12} "
                                  f"RAW={decision.top1:12} {decision.confidence*100:6.2f}% "
                                  f"OUT={decision.outcome:12} {verdict}")
                        except Exception as exc:
                            print("[CAPTURE INFERENCE ERROR]",type(exc).__name__,exc)
                        capture=None;capture_gt=None
                        rolling.clear();new_frames=0;latest_preview=None
                elif capture is None and preview_enabled and len(rolling)>=args.frames and new_frames>=args.preview_stride:
                    new_frames=0
                    try:
                        latest_preview,_,_,_=predict(list(rolling))
                    except Exception as exc:
                        print("[PREVIEW ERROR]",type(exc).__name__,exc)
                        latest_preview=None

                img=draw_landmarks(frame.copy(),res,holistic_api,draw,draw_enabled)
                if mirror:
                    img=cv2.flip(img,1)  # Draw BEFORE flip: landmark positions stay aligned.
                display=render_view(
                    img,args.width,args.height,labels,"test",latest_preview,latest_trial,
                    threshold,threshold_source,confidence_warning,fps,
                    (sample.left_present,sample.right_present,sample.pose_present,sample.face_present),
                    capture_gt if capture is not None else None,
                    args.frames-len(capture) if capture is not None else 0,
                    preview_enabled,latest_ms,len(rolling)>=args.frames,
                )
                if capture is not None and now<capture_start:
                    box_text(display,f"COUNTDOWN: {max(0,capture_start-now):.1f}s",20,55,(0,220,255),1.0,2)
                if capture is not None:
                    box_text(display,f"CAPTURE: {len(capture)}/{args.frames}",20,98,(0,220,255),.9,2)
                cv2.imshow(win,display)
                key=cv2.waitKey(1)&0xFF
                if key in (ord('q'),ord('Q'),27):
                    break
                if key in (ord('f'),ord('F')):
                    fullscreen=not fullscreen
                    cv2.setWindowProperty(win,cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow(win,args.width,args.height)
                elif key in (ord('m'),ord('M')):
                    mirror=not mirror
                elif key in (ord('p'),ord('P')):
                    preview_enabled=not preview_enabled
                    latest_preview=None
                elif key in (ord('v'),ord('V')):
                    draw_enabled=not draw_enabled
                elif key in (ord('r'),ord('R')):
                    rolling.clear();capture=None;capture_gt=None
                    new_frames=0;latest_preview=None;latest_trial=None
                    print("[RESET]")
                elif key in (ord('s'),ord('S')):
                    snapdir.mkdir(parents=True,exist_ok=True)
                    path=snapdir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.png')
                    cv2.imwrite(str(path),display)
                    print("[SNAPSHOT]",path)
                elif capture is None and (key==ord(' ') or key==ord('0') or ord('1')<=key<=ord('9')):
                    if key==ord('0'):
                        gt='UNKNOWN'
                    elif key==ord(' '):
                        gt=''
                    else:
                        idx=key-ord('1')
                        if idx>=len(labels) or labels[idx]=="UNKNOWN":
                            continue
                        gt=labels[idx]
                    capture=[]; capture_gt=gt;capture_start=now+args.countdown
                    latest_preview=None
                    rolling.clear();new_frames=0
                    print(f"[CAPTURE START] GT={gt or 'unlabeled'} in {args.countdown:.1f}s ",
                          f"for {args.frames} frames")
    finally:
        cap.release();cv2.destroyAllWindows()
        print("CSV log:",log_path)
        if not args.no_save_trials:
            print("Saved trial features:",savedir)

if __name__=='__main__':
    main()
