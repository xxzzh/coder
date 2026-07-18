from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "submission_assets" / "webui_video_capture"
OUTPUT = ROOT / "submission_assets" / "local-knowledge-webui-60s-intro.mp4"
PREVIEW_DIR = ROOT / "submission_assets" / "webui_video_previews"

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


F_TITLE = font(72, True)
F_H1 = font(54, True)
F_H2 = font(36, True)
F_BODY = font(31)
F_SMALL = font(24)
F_TAG = font(22, True)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    return (*rgb(hex_color), alpha)


def text_width(text: str, fnt: ImageFont.FreeTypeFont) -> int:
    return int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=fnt))


def wrap_text(text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if text_width(candidate, fnt) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def draw_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int] = (29, 35, 41, 255),
) -> None:
    ImageDraw.Draw(image).text(xy, text, font=fnt, fill=fill)


def draw_wrapped(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    max_width: int,
    fill: tuple[int, int, int, int] = (65, 80, 98, 255),
    line_gap: int = 10,
) -> int:
    x, y = xy
    line_height = fnt.size + line_gap
    for line in wrap_text(text, fnt, max_width):
        draw_text(image, (x, y), line, fnt, fill)
        y += line_height
    return y


def card(image: Image.Image, box: tuple[int, int, int, int], fill: str = "#ffffff", outline: str = "#d8dee6") -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box, radius=18, fill=rgba(fill, 245), outline=rgba(outline, 255), width=2)


def pill(image: Image.Image, xy: tuple[int, int], label: str, color: str = "#1769aa") -> None:
    x, y = xy
    width = text_width(label, F_TAG) + 36
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((x, y, x + width, y + 42), radius=21, fill=rgba(color, 235))
    draw_text(image, (x + 18, y + 8), label, F_TAG, (255, 255, 255, 255))


def fit_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(img, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (255, 255, 255, 0))
    canvas.alpha_composite(fitted.convert("RGBA"), ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def top_crop(img: Image.Image, height: int) -> Image.Image:
    return img.crop((0, 0, img.width, min(img.height, height)))


def add_shadowed_image(
    base: Image.Image,
    img: Image.Image,
    box: tuple[int, int, int, int],
    caption: str | None = None,
) -> None:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    shadow = Image.new("RGBA", (width + 50, height + 50), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((25, 22, width + 25, height + 22), radius=24, fill=(20, 35, 55, 68))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.alpha_composite(shadow, (x1 - 25, y1 - 20))

    panel = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width, height), radius=20, fill=255)
    fitted = fit_image(img, (width - 32, height - 32))
    panel.alpha_composite(fitted, (16, 16))
    base.paste(panel, (x1, y1), mask)
    ImageDraw.Draw(base).rounded_rectangle(box, radius=20, outline=rgba("#c9d4e2", 255), width=2)

    if caption:
        draw = ImageDraw.Draw(base)
        tw = text_width(caption, F_SMALL)
        cx = x1 + 24
        cy = y2 - 54
        draw.rounded_rectangle((cx - 10, cy - 6, cx + tw + 18, cy + 36), radius=14, fill=(255, 255, 255, 232))
        draw_text(base, (cx, cy), caption, F_SMALL, rgba("#344054", 255))


def background(t: float) -> Image.Image:
    img = Image.new("RGBA", (W, H), rgb("#f6f7f9") + (255,))
    draw = ImageDraw.Draw(img)
    for i in range(18):
        x = int((i * 181 + t * 26) % (W + 160)) - 80
        y = int(120 + (i * 73) % 820 + math.sin(t * 0.7 + i) * 18)
        draw.ellipse((x, y, x + 5, y + 5), fill=rgba("#1769aa", 55))
    draw.rectangle((0, 0, W, 9), fill=rgba("#1769aa", 255))
    draw.rectangle((0, H - 8, W, H), fill=rgba("#d8dee6", 255))
    return img


def section_header(img: Image.Image, title: str, subtitle: str, t: float, start: float, end: float) -> None:
    p = ease((t - start) / max(0.01, end - start))
    y = int(58 - (1 - p) * 22)
    draw_text(img, (84, y), title, F_H1, rgba("#1d2329", 255))
    draw_wrapped(img, (88, y + 70), subtitle, F_BODY, 760, rgba("#465667", 255), 8)


def bullets(img: Image.Image, x: int, y: int, items: list[str], color: str = "#1769aa") -> None:
    draw = ImageDraw.Draw(img)
    for item in items:
        draw.ellipse((x, y + 10, x + 14, y + 24), fill=rgba(color, 255))
        y = draw_wrapped(img, (x + 32, y), item, F_BODY, 670, rgba("#344054", 255), 7) + 14


def load_screens() -> dict[str, Image.Image]:
    files = {
        "dashboard": "01_full_dashboard.png",
        "status": "02_status_panel.png",
        "sources": "03_source_table.png",
        "operations": "04_operations_panel.png",
        "review": "05_review_panel.png",
        "api": "06_api_panel.png",
        "question": "07_question_ready.png",
        "answer": "08_answer_with_sources.png",
        "answer_panel": "09_answer_panel.png",
        "log": "10_log_panel.png",
        "guardrail": "11_no_evidence_guidance.png",
        "online": "12_online_option.png",
    }
    screens: dict[str, Image.Image] = {}
    for key, name in files.items():
        path = CAPTURE_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        screens[key] = Image.open(path).convert("RGBA")
    return screens


SCENES = [
    (0, 6, "cover"),
    (6, 13, "status"),
    (13, 20, "sources"),
    (20, 29, "question"),
    (29, 38, "answer"),
    (38, 45, "operations"),
    (45, 52, "api"),
    (52, 58, "guardrail"),
    (58, 60, "close"),
]


def scene_name(t: float) -> tuple[float, float, str]:
    for start, end, name in SCENES:
        if start <= t < end:
            return start, end, name
    return SCENES[-1]


def frame(screens: dict[str, Image.Image], idx: int) -> np.ndarray:
    t = idx / FPS
    start, end, name = scene_name(t)
    img = background(t)

    if name == "cover":
        draw_text(img, (84, 58), "Local Knowledge Base Agent", F_H1, rgba("#1d2329", 255))
        add_shadowed_image(img, screens["dashboard"], (820, 92, 1800, 888), "真实 Web UI 总览")
        pill(img, (88, 190), "本地运行")
        pill(img, (260, 190), "多格式入库")
        pill(img, (468, 190), "引用溯源")
        bullets(
            img,
            94,
            330,
            [
                "Web UI 覆盖提问、状态、来源、索引操作、OCR、API 精炼和运行日志。",
                "当前索引已包含 12 份资料、1798 个分块。",
                "资料放进来，问题问出来，答案能回到原文。",
            ],
        )
    elif name == "status":
        section_header(img, "1. 本地优先，状态清楚", "右侧状态面板直接展示资料数、文档数、分块数和待复核数量，普通用户能判断系统是否可用。", t, start, end)
        add_shadowed_image(img, screens["status"], (1020, 145, 1700, 610), "状态面板")
        card(img, (90, 340, 800, 720))
        draw_text(img, (130, 385), "加分项", F_H2, rgba("#1769aa", 255))
        bullets(img, 134, 455, ["资料和索引健康状态可视化", "OCR、离线翻译、API 可用性一屏可见", "发现待复核资料时不会静默入库"])
    elif name == "sources":
        section_header(img, "2. 多格式资料可入库", "索引来源表把文件类型、分块数量、索引状态列出来，便于解释“系统到底读了哪些资料”。", t, start, end)
        add_shadowed_image(img, screens["sources"], (770, 150, 1790, 875), "索引来源")
        bullets(img, 100, 360, ["支持 md、txt、doc、docx、pdf、xlsx", "PDF、Excel、Word 等办公资料可进入同一套检索流程", "按来源文件展示，方便排查漏导入或重复资料"])
    elif name == "question":
        section_header(img, "3. 直接提问", "不需要命令行。用户在 Web UI 输入问题，系统先查本地知识库，再整理答案和证据。", t, start, end)
        add_shadowed_image(img, screens["question"], (920, 175, 1760, 545), "提问模块")
        add_shadowed_image(img, screens["answer"], (930, 570, 1765, 900), "查询后联动状态")
        bullets(img, 100, 370, ["提问框、查询按钮和 online 开关清楚直观", "本地知识库始终优先", "流式进度和日志让等待过程可解释"])
    elif name == "answer":
        section_header(img, "4. 答案必须带引用", "命中本地资料时，答案下方列出引用文件和片段。能解释“为什么这么回答”，也便于回到原始文档核对。", t, start, end)
        add_shadowed_image(img, top_crop(screens["answer_panel"], 760), (112, 230, 1000, 900), "答案与引用")
        add_shadowed_image(img, screens["log"], (1110, 650, 1760, 900), "运行日志")
        bullets(img, 1090, 260, ["示例问题：LogMsgType 有哪些值？", "返回 STPM 框架指南作为依据", "日志记录来源类型和引用数量"])
    elif name == "operations":
        section_header(img, "5. 维护动作也在界面里", "资料更新后，用户可以直接增量更新、严格重建、跑 OCR 评测；待复核资料也有独立区域。", t, start, end)
        add_shadowed_image(img, screens["operations"], (980, 190, 1700, 575), "操作面板")
        add_shadowed_image(img, screens["review"], (980, 635, 1700, 805), "待复核资料")
        bullets(img, 110, 350, ["增量更新：新增资料快速入库", "严格重建：全量刷新，适合交付前检查", "OCR 评测：扫描件和图片文字有验证路径"])
    elif name == "api":
        section_header(img, "6. 可选增强，不绑死云端", "没有 API 也能本地检索；有合规 key 时，可用 API 精炼表达。online 只作为补充，不替代本地证据。", t, start, end)
        add_shadowed_image(img, screens["api"], (940, 125, 1688, 735), "API 精炼")
        add_shadowed_image(img, screens["online"], (930, 760, 1690, 970), "联网补充开关")
        bullets(img, 100, 355, ["本地检索、缓存和引用都在本机完成", "API 只做基于证据的总结表达", "需要联网时显式勾选 online，来源会被区分"])
    elif name == "guardrail":
        section_header(img, "7. 没依据时不编造", "当问题没有命中足够依据，界面会明确提示用户调整问法或补充资料，而不是硬凑答案。", t, start, end)
        add_shadowed_image(img, screens["guardrail"], (900, 220, 1790, 925), "无依据引导")
        bullets(img, 100, 365, ["清楚标出 source_type: none", "没有引用来源就不冒充有答案", "适合内部资料问答的可信边界"])
    else:
        draw_text(img, (120, 150), "本地知识库 Agent", F_TITLE, rgba("#1d2329", 255))
        draw_wrapped(img, (126, 260), "把分散的文档、表格、PDF 和扫描资料，变成可追溯、可维护、普通人也能使用的问答工具。", F_H1, 790, rgba("#344054", 255), 14)
        pill(img, (130, 470), "本地资料")
        pill(img, (312, 470), "混合检索")
        pill(img, (498, 470), "OCR 复核")
        pill(img, (674, 470), "引用证据")
        add_shadowed_image(img, screens["dashboard"], (990, 140, 1780, 865), "交付给用户的第一屏")
        bullets(img, 136, 610, ["项目亮点：轻量、本地、可验证、可交付", "适用：测试文档查询、内部知识问答、新人培训、资料检索"])

    return np.asarray(img.convert("RGB"))


def main() -> None:
    screens = load_screens()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        OUTPUT,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    preview_seconds = {3, 15, 25, 34, 42, 50, 56, 59}
    try:
        for idx in range(TOTAL_FRAMES):
            current = frame(screens, idx)
            writer.append_data(current)
            second = idx // FPS
            if idx % (FPS * 5) == 0:
                print(f"rendered {second:02d}s / {DURATION}s")
            if idx % FPS == 0 and second in preview_seconds:
                Image.fromarray(current).save(PREVIEW_DIR / f"frame_{second:02d}.png")
    finally:
        writer.close()

    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
