from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("tmp/pdfs/chill_math_rendered")


for topic_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
    pages = sorted(topic_dir.glob("page-*.png"))
    sheet = Image.new("RGB", (1000, 900), "#d7dde5")
    draw = ImageDraw.Draw(sheet)
    for index, page_path in enumerate(pages):
        with Image.open(page_path) as source:
            page = source.convert("RGB")
            page.thumbnail((470, 265), Image.Resampling.LANCZOS)
        x = 20 + (index % 2) * 490
        y = 25 + (index // 2) * 290
        sheet.paste(page, (x, y + 18))
        draw.text((x, y), f"Page {index + 1}", fill="#14213d")
    output = ROOT / f"contact-{topic_dir.name}.png"
    sheet.save(output)
    print(output.resolve())
