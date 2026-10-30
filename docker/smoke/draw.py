"""PIL and cairocffi draw Chinese text; numpy frames go through ffmpeg (libx264) and back through OpenCV."""
import subprocess

import cairocffi as cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# PIL with font paths that model-written code commonly hard-codes
img = Image.new('RGB', (900, 260), 'white')
d = ImageDraw.Draw(img)
for i, (path, label) in enumerate([
        ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'Noto Sans CJK 直骨门'),
        ('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', '文泉驿微米黑 直骨门'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans ok'),
        ('/usr/share/fonts/truetype/vibe-extra/lxgw-wenkai/LXGWWenKai-Regular.ttf', '霞鹜文楷 直骨门')]):
    d.text((20, 15 + i * 60), 'PIL ' + label, font=ImageFont.truetype(path, 40), fill='black')
img.save('pil.png')

# cairo toy text API resolves "sans-serif" through fontconfig (single font, no fallback)
surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 900, 120)
ctx = cairo.Context(surface)
ctx.set_source_rgb(1, 1, 1)
ctx.paint()
ctx.set_source_rgb(0, 0, 0)
ctx.select_font_face('sans-serif')
ctx.set_font_size(44)
ctx.move_to(20, 75)
ctx.show_text('cairo sans-serif 中文 直骨门')
surface.write_to_png('cairo.png')

# numpy frames -> ffmpeg libx264 -> OpenCV read-back
w, h, n = 320, 180, 30
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                       '-r', '30', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'frames.mp4'],
                      stdin=subprocess.PIPE)
for k in range(n):
    frame = np.zeros((h, w, 3), np.uint8)
    cv2.circle(frame, (20 + k * 9, h // 2), 20, (255, 120, 0), -1)
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
assert ff.wait() == 0
cap = cv2.VideoCapture('frames.mp4')
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
assert count == n, count
print(f'pil.png cairo.png frames.mp4 ({count} frames)')
