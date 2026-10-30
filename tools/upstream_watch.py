"""Weekly upstream watch (see UPSTREAM.md): tells us when Liger-Kernel ships a release that touches the files this
package forked, or whose notes mention the softmax-collapse problem. Discovery is automatic, the decision is human:
an issue labelled `upstream-sync` is opened (or commented on) until someone reviews the release and bumps
`last_reviewed_tag` in upstream.json. Never opens pull requests.

    python tools/upstream_watch.py [--dry-run]     # needs `gh` authenticated (GH_TOKEN in Actions)
"""
import argparse
import base64
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "upstream-sync"


def gh_api(path, *fields):
    cmd = ["gh", "api", path, *fields]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def fetch_file(repo, path, ref):
    data = gh_api(f"repos/{repo}/contents/{path}?ref={ref}")
    if data.get("encoding") != "base64":
        raise RuntimeError(f"{path}@{ref}: unexpected encoding {data.get('encoding')}")
    return base64.b64decode(data["content"]).decode()


def diff_stat(path, base_text, new_text):
    lines = list(difflib.unified_diff(base_text.splitlines(), new_text.splitlines(), f"{path} (base)", f"{path} (latest)", lineterm="", n=2))
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return {"path": path, "added": added, "removed": removed, "diff": "\n".join(lines) + ("\n" if lines else "")}


def keyword_hits(texts, keywords):
    """texts: {source: text}; returns [(keyword, source)] in keyword order, case-insensitive."""
    hits = []
    for kw in keywords:
        for source, text in texts.items():
            if kw.lower() in (text or "").lower():
                hits.append((kw, source))
    return hits


def should_notify(stats, hits):
    return any(s["added"] or s["removed"] for s in stats) or bool(hits)


def render_body(repo, base_tag, latest_tag, stats, commits, hits):
    out = [f"Liger-Kernel **{latest_tag}** is out and our forked files differ from the fork base **{base_tag}** "
           f"(https://github.com/{repo}/compare/{base_tag}...{latest_tag}).", "",
           "| forked file | +added | -removed |", "|---|---|---|"]
    for s in stats:
        out.append(f"| `{s['path']}` | +{s['added']} | -{s['removed']} |")
    if hits:
        out += ["", "⚠ **keyword hits** (upstream may be addressing the softmax-collapse problem itself):"]
        out += [f"- `{kw}` in {source}" for kw, source in hits]
    out += ["", "### commits touching the forked files since the base"]
    for path, items in commits.items():
        out.append(f"- `{path}`:")
        out += [f"  - {sha} {title}" for sha, title in items] or ["  - (none)"]
    out += ["", "### what to do", "Review per UPSTREAM.md (port or re-derive, then tests → bench → circus re-pin → parity) and bump "
            f"`last_reviewed_tag` to `{latest_tag}` in upstream.json. This issue stays open until then.", ""]
    for s in stats:
        if s["diff"]:
            out += [f"<details><summary>diff {s['path']} ({base_tag} → {latest_tag})</summary>", "", "```diff", s["diff"].rstrip("\n"), "```", "</details>", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args(argv)
    cfg = json.loads((ROOT / "upstream.json").read_text())
    repo, base_tag, files = cfg["repo"], cfg["base_tag"], cfg["files"]
    release = gh_api(f"repos/{repo}/releases/latest")
    latest = release["tag_name"]
    if latest == cfg["last_reviewed_tag"]:
        print(f"latest {latest} == last_reviewed_tag; nothing to do"); return 0
    base_date = gh_api(f"repos/{repo}/commits/{base_tag}")["commit"]["committer"]["date"]
    stats, commits, texts = [], {}, {"release notes": release.get("body") or ""}
    for path in files:
        base_text = fetch_file(repo, path, base_tag)
        got = hashlib.sha256(base_text.encode()).hexdigest()
        if got != cfg["base_sha256"][path]:
            raise SystemExit(f"base drift: {path}@{base_tag} sha256 {got} != recorded {cfg['base_sha256'][path]}")
        stats.append(diff_stat(path, base_text, fetch_file(repo, path, latest)))
        items = gh_api(f"repos/{repo}/commits?sha={latest}&path={path}&since={base_date}&per_page=100")
        commits[path] = [(c["sha"][:7], c["commit"]["message"].splitlines()[0]) for c in items]
        for sha, title in commits[path]:
            texts[f"commit {sha}"] = title
    hits = keyword_hits(texts, cfg["watch_keywords"])
    if not should_notify(stats, hits):
        print(f"{latest}: forked files unchanged since {base_tag}, no keyword hits"); return 0
    title = f"upstream: Liger {latest} touches forked files" + (" ⚠ keyword hit" if hits else "")
    body = render_body(repo, base_tag, latest, stats, commits, hits)
    if a.dry_run:
        print(title); print(); print(body); return 0
    subprocess.run(["gh", "label", "create", LABEL, "--force", "--color", "D93F0B", "--description", "upstream Liger release to review"], check=True, capture_output=True)
    existing = json.loads(subprocess.run(["gh", "issue", "list", "--label", LABEL, "--state", "open", "--search", f'"{latest}" in:title', "--json", "number"],
                                         check=True, capture_output=True, text=True).stdout)
    if existing:
        n = str(existing[0]["number"])
        subprocess.run(["gh", "issue", "comment", n, "--body", f"Still unreviewed as of this run.\n\n{body}"], check=True, capture_output=True)
        print(f"commented on existing issue #{n}")
    else:
        subprocess.run(["gh", "issue", "create", "--label", LABEL, "--title", title, "--body", body], check=True, capture_output=True)
        print(f"opened issue: {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
