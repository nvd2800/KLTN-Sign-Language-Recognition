"""Vietnamese UTF-8 Phase10 on-screen banner; Pillow optional on workstation.

No font files bundled. Windows installed fonts or Linux system fonts are used.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import unicodedata


@lru_cache(maxsize=4)
def _font(size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    candidates=[
        r'C:\Windows\Fonts\segoeui.ttf',
        r'C:\Windows\Fonts\arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for p in candidates:
        if Path(p).is_file():
            try:return ImageFont.truetype(p,size)
            except OSError:continue
    return None


def draw_phase10_banner(image,builder,tts_status:str, *, enabled:bool=True):
    """Draw inside left camera pane, preserving sidebar and original resolution."""
    if not enabled:return image
    import cv2
    h,w=image.shape[:2]
    left_w=min(max(480,int(w*.69)),w-360)
    panel_h=min(136,h//4)
    # Semi-opaque dark rectangle, without crop or aspect-ratio changes.
    area=image[:panel_h,:left_w].copy()
    image[:panel_h,:left_w]=cv2.addWeighted(area,.23,area*0,.77,0)
    preview=builder.preview or '(chưa có token)'
    last=builder.last_text() or '(chưa chốt câu)'
    lines=[
        'PHASE 10 | Token -> câu tiếng Việt',
        'Đang ghép: '+preview,
        'Câu gần nhất: '+last,
        'Space/N: chốt | , . ? !: dấu câu | Z: hoàn tác | X: xóa | T: đọc',
        'TTS: '+tts_status[:70],
    ]
    font=_font(19)
    if font is not None:
        from PIL import Image,ImageDraw
        rgb=cv2.cvtColor(image[:panel_h,:left_w],cv2.COLOR_BGR2RGB)
        img=Image.fromarray(rgb)
        draw=ImageDraw.Draw(img)
        for i,line in enumerate(lines):
            text=line if len(line)<78 else line[:75]+'...'
            draw.text((16,8+i*25),text,font=font,fill=(230,245,241) if i!=1 else (135,244,178))
        image[:panel_h,:left_w]=cv2.cvtColor(__import__('numpy').asarray(img),cv2.COLOR_RGB2BGR)
    else:
        # OpenCV does not render Vietnamese combining marks. Romanized preview
        # fallback is visual ONLY; transcript logs retain native Vietnamese.
        for i,line in enumerate(lines):
            ascii_line=''.join(c for c in unicodedata.normalize('NFKD',line) if not unicodedata.combining(c)).replace('đ','d').replace('Đ','D')
            cv2.putText(image,ascii_line[:72],(16,26+i*25),cv2.FONT_HERSHEY_SIMPLEX,.48,(130,245,190),1,cv2.LINE_AA)
    return image
