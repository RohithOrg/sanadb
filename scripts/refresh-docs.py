#!/usr/bin/env python3
"""Refresh the parts of the documentation that are produced by running
something, so they cannot drift from the code: example outputs, the Python
API page, and the studio screenshots. Run before a release:

    python/.venv/bin/python scripts/refresh-docs.py            # everything
    python/.venv/bin/python scripts/refresh-docs.py --examples --api
    python/.venv/bin/python scripts/refresh-docs.py --screenshots
    python/.venv/bin/python scripts/refresh-docs.py --check    # exit 1 if any output block would change

Output blocks: a Markdown page marks a fenced block with an HTML comment on
the line before it, `<!-- output: tour.py -->` or
`<!-- output: tour.py lines=2-10 -->` (1-based, inclusive); the block's
contents are replaced by those lines of that program's output. Temp paths
in the output are normalised to `/tmp/…`. The programs are listed in
PROGRAMS. The Python API page is regenerated from python/sanadb.pyi by
scripts/pyi_to_md.py. Screenshots drive the studio binary from
target/release with Playwright (python/.venv needs `playwright` and
`playwright install chromium`).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "python" / ".venv" / "bin" / "python"
DOCS = ROOT / "docs"
IMG = DOCS / "img"

# name -> (command, needs_embed_server)
PROGRAMS = {
    "tour.py": ([str(PY), "python/examples/tour.py"], False),
    "tour.rs": (["cargo", "run", "--release", "--quiet", "--example", "tour"], False),
    "failure_modes.py": ([str(PY), "python/examples/failure_modes.py"], False),
    "semantic_search.py": ([str(PY), "python/examples/semantic_search.py"], False),
    "semantic_search.rs": (
        ["cargo", "run", "--release", "--quiet", "--features", "embed-http", "--example", "semantic_search"],
        True,
    ),
}

MARK = re.compile(r"^<!-- output: (\S+)(?: lines=(\d+)-(\d+))? -->\n```text\n(.*?)```", re.M | re.S)


def normalise(text: str) -> str:
    text = re.sub(r"(/var/folders/\S+|/tmp/\S+|/private/\S+)/(docs\.sana)", r"/tmp/…/\2", text)
    text = re.sub(r"backup at \S+", "backup at /tmp/…/backup/docs.sana", text)
    text = re.sub(r"'index_path': '[^']*'", "'index_path': '…'", text)
    return text


class EmbedServer:
    def __enter__(self):
        self.p = subprocess.Popen(
            [str(PY), "python/examples/embed_server.py"], cwd=ROOT, stderr=subprocess.PIPE, text=True
        )
        for _ in range(600):  # the model loads on first start
            if self.p.poll() is not None:
                raise SystemExit("embed_server exited: " + (self.p.stderr.read() or ""))
            try:
                import urllib.request

                urllib.request.urlopen(
                    urllib.request.Request(
                        "http://127.0.0.1:8089/v1/embeddings",
                        data=b'{"input":["x"],"model":"m"}',
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=1,
                )
                return self
            except Exception:
                time.sleep(0.5)
        raise SystemExit("embed_server did not come up")

    def __exit__(self, *a):
        self.p.terminate()
        self.p.wait()


def run_programs(names):
    outputs = {}
    needs_server = any(PROGRAMS[n][1] for n in names)
    ctx = EmbedServer() if needs_server else None
    if ctx:
        ctx.__enter__()
    try:
        for n in names:
            cmd, _ = PROGRAMS[n]
            print(f"running {n}: {' '.join(cmd)}", file=sys.stderr)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"{n} failed:\n{r.stderr}")
            outputs[n] = normalise(r.stdout)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    return outputs


def splice(check: bool) -> int:
    pages = [p for p in DOCS.rglob("*.md") if MARK.search(p.read_text())]
    needed = sorted({m.group(1) for p in pages for m in MARK.finditer(p.read_text())})
    unknown = [n for n in needed if n not in PROGRAMS]
    if unknown:
        raise SystemExit(f"unknown programs in output markers: {unknown}")
    outputs = run_programs(needed)
    changed = 0
    for p in pages:
        src = p.read_text()

        def repl(m):
            name, lo, hi, old = m.group(1), m.group(2), m.group(3), m.group(4)
            lines = outputs[name].splitlines()
            if lo:
                lines = lines[int(lo) - 1 : int(hi)]
            new = "\n".join(lines).rstrip("\n") + "\n"
            head = f"<!-- output: {name}" + (f" lines={lo}-{hi}" if lo else "") + " -->\n```text\n"
            return head + new + "```"

        out = MARK.sub(repl, src)
        if out != src:
            changed += 1
            print(f"{'would change' if check else 'updated'} {p.relative_to(ROOT)}", file=sys.stderr)
            if not check:
                p.write_text(out)
    return changed


def api(check: bool) -> int:
    new = subprocess.run([sys.executable, "scripts/pyi_to_md.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    p = DOCS / "python" / "api.md"
    if p.read_text() != new:
        print(f"{'would change' if check else 'updated'} {p.relative_to(ROOT)}", file=sys.stderr)
        if not check:
            p.write_text(new)
        return 1
    return 0


def screenshots(store: str, port: int):
    from playwright.sync_api import sync_playwright

    subprocess.run(["cargo", "build", "--release", "--quiet", "-p", "sanadb-studio"], cwd=ROOT, check=True)
    env = dict(os.environ, EMBED_DIM=os.environ.get("EMBED_DIM", "128"))
    studio = subprocess.Popen(
        [
            str(ROOT / "target/release/sanadb-studio"), store, "--port", str(port),
            "--embed-cmd", f"{PY} scripts/embed_toy.py",
        ],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        url = f"http://127.0.0.1:{port}/"
        for _ in range(100):
            try:
                import urllib.request

                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                if studio.poll() is not None:
                    raise SystemExit("studio exited:\n" + studio.stdout.read())
                time.sleep(0.2)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light")
            page.goto(url)
            page.evaluate("localStorage.setItem('sanadb-studio-theme','light'); localStorage.setItem('sanadb-studio-walk-speed','instant')")
            page.reload()
            page.wait_for_function("!document.getElementById('maptitle').textContent.includes('Building')", timeout=120_000)
            page.wait_for_timeout(500)
            shot = lambda name, sel=None, clip=None: (
                page.locator(sel).first.screenshot(path=str(IMG / name)) if sel
                else page.screenshot(path=str(IMG / name), clip=clip)
            )
            # a search from a record, the whole walk drawn
            page.fill("#qid", "4242")
            page.check("#qwalk")
            page.click("#go")
            page.wait_for_selector("#hits:not([hidden]) tbody tr", timeout=60_000)
            page.wait_for_timeout(800)
            shot("studio.png")
            shot("studio-1-open.png", "header")
            shot("studio-2-query.png", "aside .sect")
            shot("studio-3-walk.png", "main")
            # the same walk in three dimensions (before the idle rotation starts)
            page.click("#view3d")
            page.wait_for_timeout(400)
            shot("studio-9-3d.png", "main")
            # the graph around a node: its out-edges and a popover with its metadata
            # (a plain search first, so the edges are not under the walk's trail)
            page.uncheck("#qwalk")
            page.click("#go")
            page.wait_for_function("document.getElementById('walkctl').hidden", timeout=60_000)
            page.evaluate("openNode(4242)")
            page.wait_for_function("graph.edges.length > 0", timeout=20_000)
            page.wait_for_timeout(400)
            shot("studio-10-graph.png", "main")
            page.evaluate("clearGraph(); hidePop()")
            page.check("#qwalk")
            page.click("#go")
            page.wait_for_selector("#walkctl:not([hidden])", timeout=60_000)
            page.wait_for_timeout(400)
            page.click("#view2d")
            page.wait_for_timeout(200)
            page.click("#hits tbody tr")
            page.wait_for_selector("#inspector .rname", timeout=10_000)
            page.wait_for_timeout(300)
            shot("studio-4-results-record.png", "section.right")
            # a filtered search: the exact-scan regime
            page.uncheck("#qwalk")
            page.fill("#qfilter", '{"cat": "7"}')
            page.click("#go")
            page.wait_for_function("document.getElementById('mode').textContent.includes('filtered')", timeout=60_000)
            page.wait_for_timeout(300)
            shot("studio-5-filtered.png", "section.right .results")
            page.fill("#qfilter", "")
            # cluster labels
            page.select_option("#labelkey", "lang")
            page.wait_for_timeout(1500)
            shot("studio-6-labels.png", "main")
            page.select_option("#labelkey", "")
            # files tab
            page.get_by_role("tab", name="Files").click()
            page.wait_for_timeout(300)
            shot("studio-7-files.png", "aside")
            page.get_by_role("tab", name="Records").click()
            # text search through the attached embedding command
            page.fill("#qid", "")
            page.fill("#qembed", "a photograph of a harbour at dusk")
            page.click("#go")
            page.wait_for_selector("#hits:not([hidden]) tbody tr", timeout=60_000)
            page.wait_for_timeout(300)
            shot("studio-8-embed.png", clip={"x": 0, "y": 0, "width": 740, "height": 420})
            # dark chrome
            page.click("#theme")
            page.wait_for_timeout(500)
            shot("studio-dark.png")
            b.close()
    finally:
        studio.terminate()
        studio.wait()
    print("screenshots written to docs/img", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--screenshots", action="store_true")
    ap.add_argument("--check", action="store_true", help="report what --examples and --api would change; exit 1 if anything")
    ap.add_argument("--store", default="data/sift/store/sift1000.labels.sana", help="store for the screenshots")
    ap.add_argument("--port", type=int, default=7071)
    a = ap.parse_args()
    everything = not (a.examples or a.api or a.screenshots)
    changed = 0
    if a.api or everything or a.check:
        changed += api(a.check)
    if a.examples or everything or a.check:
        changed += splice(a.check)
    if (a.screenshots or everything) and not a.check:
        screenshots(a.store, a.port)
    if a.check and changed:
        print(f"{changed} page(s) out of date; run scripts/refresh-docs.py", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
