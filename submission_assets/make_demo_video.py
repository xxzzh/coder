from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "submission_assets"
WALLPAPER = OUT_DIR / "local-knowledge-wallpaper-punk.png"
OUTPUT = OUT_DIR / "local-knowledge-60s-demo-1080p.mp4"

W, H = 1920, 1080
FPS = 24
DURATION = 60
TOTAL_FRAMES = FPS * DURATION


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(90, True)
F_H1 = font(64, True)
F_H2 = font(42, True)
F_BODY = font(34)
F_SMALL = font(26)
F_MONO = font(30, True)


def cover_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    sw, sh = img.size
    tw, th = size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def scene_progress(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


def alpha_layer() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def draw_text(
    layer: Image.Image,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int] = (255, 255, 255, 255),
    stroke: int = 0,
) -> None:
    d = ImageDraw.Draw(layer)
    d.text(xy, text, font=fnt, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0, 190))


def card(
    layer: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int] = (8, 18, 34, 185),
    outline: tuple[int, int, int, int] = (0, 210, 255, 180),
    radius: int = 28,
    width: int = 2,
) -> None:
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def neon_line(layer: Image.Image, points: list[tuple[int, int]], fill: tuple[int, int, int, int]) -> None:
    d = ImageDraw.Draw(layer)
    for width, alpha in [(12, 42), (6, 88), (2, fill[3])]:
        color = (fill[0], fill[1], fill[2], alpha)
        d.line(points, fill=color, width=width, joint="curve")


def pill(layer: Image.Image, box: tuple[int, int, int, int], text: str, color: str) -> None:
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=22, fill=rgba(color, 210), outline=rgba("#ffffff", 95), width=1)
    tw = d.textlength(text, font=F_SMALL)
    draw_text(layer, (int((box[0] + box[2] - tw) / 2), box[1] + 8), text, F_SMALL)


def wrap_lines(text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(candidate, font=fnt) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def draw_bullets(layer: Image.Image, x: int, y: int, lines: list[str], color: str = "#21d7ff") -> None:
    for i, text in enumerate(lines):
        yy = y + i * 56
        d = ImageDraw.Draw(layer)
        d.ellipse((x, yy + 11, x + 16, yy + 27), fill=rgba(color, 255))
        draw_text(layer, (x + 34, yy), text, F_BODY)


def draw_window(layer: Image.Image, box: tuple[int, int, int, int], title: str) -> None:
    x1, y1, x2, y2 = box
    card(layer, box, fill=(5, 12, 24, 214), outline=(42, 213, 255, 170), radius=26)
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x1, y1, x2, y1 + 66), radius=26, fill=(9, 28, 48, 230))
    d.rectangle((x1, y1 + 38, x2, y1 + 66), fill=(9, 28, 48, 230))
    for i, c in enumerate(["#ff4f9a", "#22d7ff", "#a1ff25"]):
        d.ellipse((x1 + 22 + i * 32, y1 + 22, x1 + 40 + i * 32, y1 + 40), fill=rgba(c, 255))
    draw_text(layer, (x1 + 128, y1 + 16), title, F_SMALL, fill=(220, 246, 255, 240))


def bg_frame(base: Image.Image, t: float) -> Image.Image:
    shift = int(math.sin(t * 0.6) * 16)
    bg = base.transform((W, H), Image.Transform.AFFINE, (1, 0, shift, 0, 1, 0), Image.Resampling.BICUBIC)
    bg = ImageEnhance.Contrast(bg).enhance(1.08)
    bg = ImageEnhance.Brightness(bg).enhance(0.72)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 80))
    img = bg.convert("RGBA")
    img.alpha_composite(overlay)
    return img


def draw_scene_intro(layer: Image.Image, t: float) -> None:
    p = scene_progress(t, 0, 6)
    dx = int((1 - p) * -130)
    draw_text(layer, (120 + dx, 160), "本地知识库", F_TITLE, stroke=3)
    draw_text(layer, (126 + dx, 280), "60 秒演示：从资料到可追溯答案", F_H2, fill=(235, 253, 255, 235))
    pill(layer, (128, 365, 328, 420), "本地 RAG", "#02c8ff")
    pill(layer, (356, 365, 584, 420), "引用溯源", "#ff2ccf")
    pill(layer, (612, 365, 842, 420), "无命中引导", "#9cff1b")
    draw_bullets(
        layer,
        136,
        500,
        ["支持 txt / doc / docx / xlsx / pdf / md", "可本地运行，可选 API 精炼和联网补充", "适合测试文档查询、内部资料问答、办公知识管理"],
    )


def draw_scene_start(layer: Image.Image, t: float) -> None:
    p = scene_progress(t, 6, 13)
    draw_text(layer, (90, 82), "第一步：打开工具，放入资料", F_H1)
    draw_window(layer, (98, 210, 910, 850), "start.bat 一键菜单")
    items = [
        "1  首次配置环境",
        "2  启动 Web UI",
        "3  打开资料目录 knowledge_base/raw",
        "4  严格重建索引",
        "5  健康检查",
    ]
    for i, item in enumerate(items):
        y = 310 + i * 82
        card(layer, (148, y, 852, y + 58), fill=(14, 35, 58, 195), outline=(255, 255, 255, 35), radius=16, width=1)
        draw_text(layer, (174, y + 10), item, F_BODY, fill=(236, 250, 255, int(255 * p)))
    draw_window(layer, (1030, 240, 1780, 820), "knowledge_base/raw")
    for i, name in enumerate(["markdown", "word", "pdf", "excel"]):
        x = 1090 + (i % 2) * 330
        y = 350 + (i // 2) * 170
        card(layer, (x, y, x + 250, y + 110), fill=(9, 24, 42, 210), outline=(35, 210, 255, 160), radius=22)
        draw_text(layer, (x + 38, y + 34), name, F_BODY)
    neon_line(layer, [(920, 530), (1000, 530), (1028, 530)], (255, 42, 209, 230))


def draw_scene_ingest(layer: Image.Image, t: float) -> None:
    p = scene_progress(t, 13, 20)
    draw_text(layer, (90, 82), "第二步：抽取、OCR、分块、索引", F_H1)
    steps = [
        ("资料导入", "PDF / Excel / Word / Markdown"),
        ("OCR 抽取", "扫描件和图片文字可复核"),
        ("语义分块", "保留来源、页码、行列信息"),
        ("混合检索", "关键词 + 向量 + 规则增强"),
    ]
    start_x = 140
    for i, (title, desc) in enumerate(steps):
        x = start_x + i * 440
        y = 355 + int(math.sin(t * 2 + i) * 10)
        card(layer, (x, y, x + 345, y + 220), fill=(5, 18, 36, 218), outline=(0, 220, 255, 160), radius=28)
        draw_text(layer, (x + 38, y + 36), title, F_H2)
        for j, line in enumerate(wrap_lines(desc, F_SMALL, 270)):
            draw_text(layer, (x + 38, y + 116 + j * 34), line, F_SMALL, fill=(224, 246, 255, 230))
        if i < len(steps) - 1:
            neon_line(layer, [(x + 360, y + 110), (x + 424, y + 110)], (156, 255, 30, int(230 * p)))
    draw_text(layer, (160, 690), "增量更新可快速补充资料，严格重建用于全量刷新索引。", F_BODY, fill=(230, 247, 255, 235))


def draw_scene_qa(layer: Image.Image, t: float) -> None:
    draw_text(layer, (90, 82), "第三步：在 Web UI 直接提问", F_H1)
    draw_window(layer, (120, 190, 1800, 900), "Local Knowledge Base Agent")
    card(layer, (170, 285, 1120, 470), fill=(255, 255, 255, 238), outline=(0, 208, 255, 140), radius=22)
    query = "HPD test怎么配置？"
    typed = query[: max(1, min(len(query), int((t - 20) * 4)))]
    draw_text(layer, (215, 340), typed, F_H2, fill=(6, 18, 32, 255))
    card(layer, (170, 525, 1120, 820), fill=(255, 255, 255, 238), outline=(255, 45, 205, 120), radius=22)
    y = 575
    answer_lines = [
        "根据本地 PDF 资料，系统整理出 HPD test 的配置要点。",
        "同时标出引用来源，方便核对原文。",
    ]
    for line in answer_lines:
        draw_text(layer, (215, y), line, F_BODY, fill=(8, 22, 36, 255))
        y += 50
    for i, label in enumerate(["本地知识库", "cache", "api"]):
        pill(layer, (1265, 330 + i * 88, 1505, 386 + i * 88), label, ["#02c8ff", "#9cff1b", "#ff2ccf"][i])
    draw_text(layer, (1260, 635), "状态标签显示答案来源", F_BODY)
    draw_text(layer, (1260, 690), "避免把缓存、API、本地检索混在一起。", F_SMALL, fill=(220, 246, 255, 220))


def draw_scene_sources(layer: Image.Image, t: float) -> None:
    draw_text(layer, (90, 82), "第四步：答案必须能追到来源", F_H1)
    draw_window(layer, (105, 190, 900, 850), "答案")
    answer_lines = [
        "HPD Test 需要编辑 HPD 测试配置文件。",
        "设置结果目录、BOM 目录和测试周期。",
        "PresentTest=1 表示检测到人体存在即通过。",
    ]
    for i, line in enumerate(answer_lines):
        draw_text(layer, (150, 300 + i * 82), line, F_BODY)
    draw_window(layer, (1000, 250, 1790, 820), "引用来源")
    sources = [
        "PDF：HPD Human Presence Detection Test",
        "页码：4.2 HPD Test",
        "片段：TestTimeInterval / PresentTest",
    ]
    for i, s in enumerate(sources):
        card(layer, (1050, 340 + i * 115, 1740, 420 + i * 115), fill=(14, 35, 58, 220), outline=(255, 255, 255, 45), radius=18, width=1)
        draw_text(layer, (1085, 360 + i * 115), s, F_BODY)
    neon_line(layer, [(870, 430), (980, 430), (1008, 430)], (255, 42, 209, 220))


def draw_scene_examples(layer: Image.Image, t: float) -> None:
    draw_text(layer, (90, 82), "覆盖真实查询场景", F_H1)
    examples = [
        ("SR是什么意思", "缩写词解释命中本地词典"),
        ("HPD test", "PDF 内容可检索"),
        ("LogMsgType有哪些值", "表格/枚举值可回答"),
        ("没有依据的问题", "明确提示并给出下一步"),
    ]
    for i, (q, a) in enumerate(examples):
        x = 170 + (i % 2) * 800
        y = 250 + (i // 2) * 250
        card(layer, (x, y, x + 650, y + 170), fill=(5, 18, 36, 218), outline=(0, 220, 255, 150), radius=26)
        draw_text(layer, (x + 36, y + 30), q, F_H2)
        draw_text(layer, (x + 38, y + 98), a, F_SMALL, fill=(220, 246, 255, 225))
    draw_text(layer, (170, 810), "不是只做通用聊天，而是围绕本地资料给出可核对答案。", F_BODY)


def draw_scene_guardrail(layer: Image.Image, t: float) -> None:
    draw_text(layer, (90, 82), "没有命中时，也要给用户下一步", F_H1)
    draw_window(layer, (130, 210, 1740, 850), "无命中引导")
    card(layer, (190, 310, 1120, 455), fill=(255, 255, 255, 238), outline=(0, 208, 255, 120), radius=20)
    draw_text(layer, (235, 352), "ZZZ999有哪些测试项？", F_H2, fill=(5, 18, 32, 255))
    card(layer, (190, 515, 1120, 650), fill=(255, 255, 255, 238), outline=(255, 255, 255, 60), radius=20)
    draw_text(layer, (235, 558), "本地知识库没有找到足够依据。", F_BODY, fill=(5, 18, 32, 255))
    draw_text(
        layer,
        (235, 735),
        "本地索引存在，但当前问法无法命中足够依据。请换成已有索引能回答的问题。",
        F_BODY,
        fill=(255, 58, 58, 255),
    )
    draw_text(layer, (1210, 365), "避免编造", F_H2, fill=(255, 255, 255, 255))
    draw_text(layer, (1210, 430), "未知对象不硬凑建议；\n已验证可答的问题才提示用户。", F_BODY, fill=(230, 247, 255, 230))


def draw_scene_optional(layer: Image.Image, t: float) -> None:
    draw_text(layer, (90, 82), "可选：API 精炼与联网补充", F_H1)
    blocks = [
        ("本地优先", "先检索本地索引，答案必须围绕证据。", "#02c8ff"),
        ("API 精炼", "有 key 时用大模型整理表达，不替代本地依据。", "#ff2ccf"),
        ("联网补充", "勾选 online 时补充外部结果，并标注来源。", "#9cff1b"),
    ]
    for i, (title, desc, color) in enumerate(blocks):
        x = 160 + i * 560
        card(layer, (x, 330, x + 460, 620), fill=(6, 18, 34, 222), outline=rgba(color, 190), radius=30)
        draw_text(layer, (x + 46, 385), title, F_H2)
        yy = 470
        for line in wrap_lines(desc, F_BODY, 360):
            draw_text(layer, (x + 46, yy), line, F_BODY, fill=(226, 246, 255, 230))
            yy += 48
    draw_text(layer, (175, 735), "敏感资料不需要上传建库；无 API key 也能纯本地使用。", F_BODY, fill=(235, 250, 255, 240))


def draw_scene_close(layer: Image.Image, t: float) -> None:
    draw_text(layer, (120, 150), "本地知识库", F_TITLE, stroke=3)
    draw_text(layer, (126, 275), "把分散文档变成可追溯、可验证、可提问的知识资产", F_H2)
    draw_bullets(
        layer,
        136,
        430,
        ["资料留在本地，适合内部文档和测试资料", "OCR、多格式导入、混合检索、增量索引", "答案带引用，没依据时不编造"],
        color="#ff2ccf",
    )
    card(layer, (1180, 360, 1740, 640), fill=(5, 18, 36, 220), outline=(0, 220, 255, 170), radius=28)
    draw_text(layer, (1235, 410), "适用场景", F_H1)
    draw_text(layer, (1240, 515), "资料问答 · 新人培训 · 测试文档查询", F_BODY, fill=(225, 248, 255, 230))


SCENES = [
    (0, 6, draw_scene_intro),
    (6, 13, draw_scene_start),
    (13, 20, draw_scene_ingest),
    (20, 28, draw_scene_qa),
    (28, 36, draw_scene_sources),
    (36, 44, draw_scene_examples),
    (44, 51, draw_scene_guardrail),
    (51, 56, draw_scene_optional),
    (56, 60, draw_scene_close),
]


def active_scene(t: float):
    for start, end, fn in SCENES:
        if start <= t < end:
            return start, end, fn
    return SCENES[-1]


def make_frame(base: Image.Image, frame_index: int) -> np.ndarray:
    t = frame_index / FPS
    img = bg_frame(base, t)
    layer = alpha_layer()

    d = ImageDraw.Draw(layer)
    for i in range(16):
        x = int((i * 173 + t * 42) % (W + 220)) - 120
        y = int((i * 71 + math.sin(t + i) * 35) % H)
        d.ellipse((x, y, x + 4, y + 4), fill=(0, 220, 255, 120))
    neon_line(layer, [(-80, 108), (420, 70), (850, 120)], (255, 42, 209, 110))
    neon_line(layer, [(1260, 980), (1600, 900), (2050, 920)], (0, 220, 255, 120))

    start, end, fn = active_scene(t)
    local_p = scene_progress(t, start, min(start + 1.0, end))
    content = alpha_layer()
    fn(content, t)
    if local_p < 1:
        content.putalpha(ImageEnhance.Brightness(content.getchannel("A")).enhance(local_p))
    layer.alpha_composite(content)

    img.alpha_composite(layer)
    return np.asarray(img.convert("RGB"))


def main() -> None:
    if not WALLPAPER.exists():
        raise FileNotFoundError(WALLPAPER)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = cover_resize(Image.open(WALLPAPER).convert("RGB"), (W, H)).filter(ImageFilter.GaussianBlur(1.2))

    writer = imageio.get_writer(
        OUTPUT,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame_index in range(TOTAL_FRAMES):
            writer.append_data(make_frame(base, frame_index))
            if frame_index % (FPS * 5) == 0:
                print(f"rendered {frame_index // FPS:02d}s / {DURATION}s")
    finally:
        writer.close()

    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
