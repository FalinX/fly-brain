import pathlib
from playwright.sync_api import sync_playwright
src = pathlib.Path("report.html").resolve().as_uri()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1123, "height": 794})
    pg.goto(src, wait_until="networkidle"); pg.wait_for_timeout(2500)
    rows = pg.evaluate("""() => [...document.querySelectorAll('.page')].map((el,i)=>{
        const pad = parseFloat(getComputedStyle(el).paddingBottom);
        let bottom = 0;
        for (const c of el.children){
          if (c.classList.contains('rail')||c.classList.contains('folio')||
              c.classList.contains('bleed')||c.classList.contains('veil')||
              c.classList.contains('num')||c.classList.contains('inner')) continue;
          const r = c.getBoundingClientRect(), pr = el.getBoundingClientRect();
          bottom = Math.max(bottom, r.bottom - pr.top);
        }
        return {i, h: Math.round(el.getBoundingClientRect().height),
                content: Math.round(bottom), pad: Math.round(pad)};
      })""")
    for r in rows:
        limit = r["h"] - r["pad"]
        flag = "OVERFLOW" if r["content"] > limit + 1 else ("tight" if r["content"] > limit - 20 else "ok")
        print(f"page {r['i']:2d}  content {r['content']:4d} / limit {limit:4d}   {flag}")
    b.close()
