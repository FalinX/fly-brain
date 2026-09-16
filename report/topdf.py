import pathlib
from playwright.sync_api import sync_playwright

src = pathlib.Path("report.html").resolve().as_uri()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1123, "height": 794})
    pg.goto(src, wait_until="networkidle")
    pg.wait_for_timeout(2500)
    pg.evaluate("document.fonts.ready")
    pg.pdf(path="fly-ai-explained-TH.pdf", width="297mm", height="210mm",
           print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
print("pdf written")
