"""Review *actual* test mistakes; this script never edits training data.

python scripts/review_v3_test_errors.py --predictions artifacts/experiments/tcn_v3_spatial/predictions_test.csv
"""
from __future__ import annotations
import argparse,json,csv,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.v3_eval_trace import read_csv_rows,write_csv

REVIEW_COLUMNS=['sample_id','subject','condition','reference_sign','true_label','predicted_label',
                'confidence','error_type','main_hand_valid_ratio','pose_valid_ratio','qc_status',
                'video_path','raw_path','human_confirmed_label','human_review_notes','review_status']


def summarize(rows):
    errors=[r for r in rows if str(r.get('true_label',''))!=str(r.get('predicted_label',''))]
    unk=[r for r in errors if r.get('true_label')=='UNKNOWN']
    token_risk=[r for r in unk if r.get('predicted_label') not in ('UNKNOWN','REST')]
    rest=[r for r in unk if r.get('predicted_label')=='REST']
    return {'n_predictions':len(rows),'n_errors':len(errors),'unknown_errors':len(unk),
            'unknown_to_rest':len(rest),'unknown_to_token_risk':len(token_risk),
            'by_error_type':dict(Counter(r.get('error_type','') for r in errors)),
            'by_condition':dict(Counter(r.get('condition','UNRESOLVED') for r in unk)),
            'unresolved_provenance':sum(r.get('source_type','')=='UNRESOLVED' for r in errors),
            'missing_source_video':sum(not r.get('video_path') for r in errors),
            'notes':['These are original TEST errors, not automatically valid TRAIN hard negatives.',
                     'Review source videos/landmark QC manually before attributing a cause.',
                     'A predicted REST is a classification mistake but not a sign-token emission.',
                     'Do not train V3.1 using this TEST material or tune thresholds on it.']}


def review(rows):
    return [dict((k,r.get(k,'')) for k in REVIEW_COLUMNS)|{'review_status':'PENDING_HUMAN_REVIEW'}
            for r in rows if str(r.get('true_label',''))!=str(r.get('predicted_label',''))]


def render_markdown(stats, rows):
    lines=['# TEST Error Review – TCN V3 Spatial 513D','',
           f"- Số sequence TEST: {stats['n_predictions']}",
           f"- Số mẫu nhận sai: {stats['n_errors']}",
           f"- UNKNOWN → REST: {stats['unknown_to_rest']} (không trực tiếp phát token)",
           f"- UNKNOWN → ký hiệu có nghĩa: {stats['unknown_to_token_risk']} (nguy cơ phát token)",
           '', '| Sample ID | Signer | Condition | Actual | Predicted | Confidence | QC | Video |',
           '|---|---|---|---|---|---:|---|---|']
    for r in rows:
        p=r.get('video_path','') or 'CHƯA CÓ VIDEO'
        lines.append('| ' + ' | '.join(str(x).replace('|','/') for x in (
            r.get('sample_id'),r.get('subject'),r.get('condition'),r.get('true_label'),
            r.get('predicted_label'),r.get('confidence'),r.get('qc_status'),p))+' |')
    lines += ['', '## Quy trình xác nhận thủ công',
              '1. Mở source_video.avi nếu có; đối chiếu ký hiệu, vị trí tay, hướng, và nhãn UNKNOWN.',
              '2. Kiểm tra left/right/pose validity; mẫu ít landmark phải được gắn QC REVIEW, không tự động loại.',
              '3. Điền human_confirmed_label / human_review_notes vào error_review.csv.',
              '4. Chỉ thu mẫu *mới từ TRAIN signers* cùng loại lỗi. Không đưa các file TEST này vào TRAIN.',
              '5. Kết quả sau khi sửa trên TEST cũ chỉ là exploratory; dùng một BLIND TEST mới để báo cáo kết luận.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',default='artifacts/experiments/tcn_v3_spatial/predictions_test.csv')
    p.add_argument('--out',default='logs/v3_test_error_review')
    a=p.parse_args()
    def rp(x):
        q=Path(x);return q if q.is_absolute() else ROOT/q
    rows=read_csv_rows(rp(a.predictions))
    if not rows:raise RuntimeError(f'Prediction CSV missing or empty: {rp(a.predictions)}. Run evaluate_v3_spatial.py --split test')
    errors=review(rows);s=summarize(rows)
    out=rp(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'error_review.csv',errors,columns=REVIEW_COLUMNS)
    (out/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README_review.md').write_text(render_markdown(s,errors),encoding='utf-8')
    print(json.dumps(s,indent=2,ensure_ascii=False))
    print('Review files:',out)

if __name__=='__main__':main()
