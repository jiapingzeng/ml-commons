# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fetches the public sources of the real corpus and builds <work>/corpus_real.json.

Sources, pinned in data/real_corpus_manifest.json:
  - opensearch-project/documentation-website at a fixed commit: _ml-commons-plugin, _search-plugins, _vector-search
    and _api-reference/{search-apis,document-apis}; one document per Markdown file, split at headings into passages
    of at most about 220 words. Date = the file's last commit date at that commit.
  - opensearch-project/opensearch-build at a fixed commit: release-notes/opensearch-release-notes-<v>.md for 2.10.0
    and later (no alpha/beta/rc): Release Highlights plus the FEATURES / ENHANCEMENTS / BUG FIXES / BREAKING CHANGES
    sections of the search and ML components. Date = the date the file was added to the repository.
  - A fixed list of GitHub issues of ml-commons, neural-search and k-NN (GitHub search API): status unofficial; each
    issue keeps at most its first two passages. Issue text can be edited after the snapshot, so these passages may
    differ slightly; no question has its answer in an issue.
  - The OpenSearch Dashboards sample data sets (logs, flights, ecommerce) at a fixed commit, first 600 documents each,
    written to <work>/src_real/ for the unrelated sample indices.
Every `${parameters` in passage text is written as `$ {parameters`: ml-commons rejects any LLM request payload that
contains `${parameters` (Connector.validatePayload), and connector examples in the docs contain it.
At the end, every gold passage of the real questions is checked against the manifest's text hashes.

Needs git and network access to github.com. Set GITHUB_TOKEN to raise the GitHub search API rate limit (unauthenticated
requests are limited to 10 per minute, which this script respects).
Usage: python3 fetch_real_corpus.py [--skip-clone]"""

import collections
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import requests

from common import DATA, Config

MAX_WORDS = 220
MIN_WORDS = 40
DOC_SETS = {  # source index -> (doc_type, folders)
    "docs-ml-commons": ("ml-commons-docs", ["_ml-commons-plugin"]),
    "docs-search": ("search-docs", ["_search-plugins"]),
    "docs-vector-search": ("vector-search-docs", ["_vector-search"]),
    "docs-api-reference": ("api-reference", ["_api-reference/search-apis", "_api-reference/document-apis"]),
}
RN_COMPONENTS = ("ML Commons", "Neural Search", "k-NN", "Skills", "Flow Framework", "Search Relevance")
RN_CATEGORIES = ("FEATURES", "ENHANCEMENTS", "BUG FIXES", "BREAKING CHANGES")
ID_PREFIX = {"docs-ml-commons": "mlc", "docs-search": "srch", "docs-vector-search": "vec", "docs-api-reference": "api"}
SAMPLE = {"logs": "logs/logs.json.gz", "flights": "flights/flights.json.gz", "ecommerce": "ecommerce/ecommerce.json.gz"}
SAMPLE_PATH = "src/plugins/home/server/services/sample_data/data_sets/"
SAMPLE_DOCS = 600


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def clone(url, path, sha, sparse=None):
    """Blobless clone checked out at sha (history stays available for the date lookups)."""
    if not os.path.isdir(os.path.join(path, ".git")):
        git("clone", "--filter=blob:none", "--no-checkout", url, path)
    if sparse:
        git("sparse-checkout", "set", *sparse, cwd=path)
    git("checkout", "--quiet", sha, cwd=path)


def git_dates(repo, rev, paths, added=False):
    """path -> date: last change (or first add with added=True) at rev, from one git log pass."""
    args = ["log", "--format=@%as", "--name-only"]
    if added:
        args += ["--diff-filter=A", "--reverse"]
    out = git(*args, rev, "--", *paths, cwd=repo)
    dates, cur = {}, None
    for line in out.splitlines():
        if line.startswith("@"):
            cur = line[1:]
        elif line.strip() and line not in dates:
            dates[line] = cur
    return dates


def front_matter(text):
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        for line in text[3:end].splitlines():
            m = re.match(r"^(\w+):\s*(.*)$", line)
            if m:
                meta[m.group(1)] = m.group(2).strip().strip('"')
        text = text[end + 4:]
    return meta, text


def clean_md(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\{%-?.*?-?%\}", "", text, flags=re.S)
    text = re.sub(r"^\{:[^}]*\}\s*$", "", text, flags=re.M)
    text = re.sub(r"\{:[^}]*\}", "", text)
    text = re.sub(r"\[([^\]]+)\]\((?:\{\{[^)]*\}\}|[^)])*\)", r"\1", text)  # links -> text
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"</?(details|summary|p|div|span|br)[^>]*>", "", text)
    return text


def sections(text):
    """[(heading, body)] split at Markdown headings outside code fences."""
    out, heading, buf, fence = [], "", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            if "".join(buf).strip():
                out.append((heading, "\n".join(buf).strip()))
            heading, buf = re.sub(r"[`*]", "", m.group(2)).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((heading, "\n".join(buf).strip()))
    return out


def split_words(body, max_words=MAX_WORDS):
    """Splits at paragraph boundaries into pieces of at most max_words (long paragraphs are cut by words)."""
    units = []
    for para in re.split(r"\n\s*\n", body):
        para = "\n".join(x.rstrip() for x in para.strip().splitlines() if x.strip())
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append((para, "\n\n"))
            continue
        for line in para.splitlines():
            words = line.split()
            while len(words) > max_words:
                units.append((" ".join(words[:max_words]), "\n"))
                words = words[max_words:]
            if words:
                units.append((" ".join(words), "\n"))
    pieces, cur, n = [], "", 0
    for text, sep in units:
        w = len(text.split())
        if cur and n + w > max_words:
            pieces.append(cur)
            cur, n = "", 0
        cur = cur + sep + text if cur else text
        n += w
    if cur:
        pieces.append(cur)
    return pieces


def chunk_sections(secs):
    """Merges short sections forward, then splits long ones; returns [(heading, text)]."""
    merged, pend_h, pend = [], None, ""
    for h, b in secs:
        if pend_h is None:
            pend_h = h
        pend = (pend + "\n\n" + (f"{h}\n" if pend and h else "") + b).strip()
        if len(pend.split()) >= MIN_WORDS:
            merged.append((pend_h, pend))
            pend_h, pend = None, ""
    if pend:
        if merged and len(pend.split()) < MIN_WORDS:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + pend)
        else:
            merged.append((pend_h or "", pend))
    out = []
    for h, b in merged:
        for piece in split_words(b):
            out.append((h, piece))
    return out


def slug(s, n=60):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n].strip("-")


def docs_passages(docs, rev):
    out = []
    folders = [f for _, fs in DOC_SETS.values() for f in fs]
    dates = git_dates(docs, rev, folders)
    for index, (doc_type, fs) in DOC_SETS.items():
        for folder in fs:
            for dp, _, files in sorted(os.walk(os.path.join(docs, folder))):
                for fn in sorted(files):
                    if not fn.endswith(".md"):
                        continue
                    path = os.path.relpath(os.path.join(dp, fn), docs)
                    meta, text = front_matter(open(os.path.join(docs, path), errors="ignore").read())
                    title = meta.get("title") or fn[:-3]
                    secs = sections(clean_md(text))
                    secs = [(h if h != title else "", b) for h, b in secs]
                    rel = path.split("/", 1)[1][:-3]
                    if rel.endswith("/index"):
                        rel = rel[:-6] or "index"
                    doc = f"{ID_PREFIX[index]}-{slug(rel)}"
                    permalink = meta.get("permalink") or "/" + path.split("/", 1)[0].lstrip("_") + "/" + rel + "/"
                    for i, (h, t) in enumerate(chunk_sections(secs), 1):
                        out.append({"id": f"{doc}-c{i}", "doc": doc, "index": index, "doc_type": doc_type,
                                    "status": "official", "title": title, "section": h, "text": t,
                                    "parent": meta.get("parent", ""), "grand_parent": meta.get("grand_parent", ""),
                                    "path": path, "url": "https://docs.opensearch.org/latest" + permalink,
                                    "date": dates.get(path), "chunk": i})
    return out


def version_key(v):
    return tuple(int(x) for x in v.split("."))


def release_passages(build, rev):
    out = []
    files = {}
    for fn in os.listdir(os.path.join(build, "release-notes")):
        m = re.match(r"opensearch-release-notes-(\d+\.\d+\.\d+)\.md$", fn)
        if m and version_key(m.group(1)) >= (2, 10, 0):
            files[m.group(1)] = "release-notes/" + fn
    added = git_dates(build, rev, ["release-notes"], added=True)
    for v in sorted(files, key=version_key):
        path = files[v]
        lines = clean_md(open(os.path.join(build, path)).read()).splitlines()
        blocks, cat, comp, buf = [], None, None, []

        def flush():
            body = "\n".join(buf).strip()
            if body and cat and (cat == "Release Highlights" or (cat in RN_CATEGORIES and comp and
                                                                  any(c in comp for c in RN_COMPONENTS))):
                blocks.append((cat, comp, body))

        for line in lines:
            m2 = re.match(r"^##\s+(.*)$", line)
            m3 = re.match(r"^###\s+(.*)$", line)
            if m2:
                flush()
                cat, comp, buf = m2.group(1).strip(), None, []
                cat = "Release Highlights" if cat.lower().startswith("release highlights") else cat.upper()
            elif m3 and cat != "Release Highlights":
                flush()
                comp, buf = m3.group(1).strip(), []
            else:
                buf.append(line)
        flush()
        doc = f"rn-{v.replace('.', '-')}"
        n = 0
        for cat, comp, body in blocks:
            body = re.sub(r"\s*\(\[?#?(\d+)\]?\)", r" (#\1)", body)
            heading = "Release highlights" if cat == "Release Highlights" else f"{cat.title()}: {comp}"
            for piece in split_words(body):
                n += 1
                out.append({"id": f"{doc}-c{n}", "doc": doc, "index": "release-notes", "doc_type": "release-notes",
                            "status": "official", "title": f"OpenSearch {v} release notes", "section": heading,
                            "text": piece, "version": v, "category": cat.title(),
                            "component": "" if cat == "Release Highlights" else comp.replace("OpenSearch ", ""),
                            "path": path, "url": f"https://github.com/opensearch-project/opensearch-build/blob/main/{path}",
                            "date": added.get(path), "chunk": n})
    return out


def github(url, params):
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    for attempt in range(8):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code in (403, 429):  # rate limited: wait for the window to reset
            reset = int(r.headers.get("x-ratelimit-reset", time.time() + 60))
            time.sleep(max(5, min(120, reset - time.time() + 2)))
            continue
        r.raise_for_status()
        time.sleep(1 if os.environ.get("GITHUB_TOKEN") else 7)
        return r.json()
    raise RuntimeError(f"GitHub API kept rate limiting {url}")


def fetch_issues(manifest, out_path):
    """The manifest's issues, in manifest order, via the search API (one created-date window per repository)."""
    if os.path.exists(out_path):
        return json.load(open(out_path))
    wanted = collections.defaultdict(set)
    for repo, number in manifest["issues"]["list"]:
        wanted[repo].add(number)
    found = {}
    for repo, (lo, hi) in manifest["issues"]["created"].items():
        page = 1
        while True:
            j = github("https://api.github.com/search/issues",
                       {"q": f"repo:opensearch-project/{repo} is:issue created:{lo}..{hi}", "sort": "created",
                        "order": "asc", "per_page": 100, "page": page})
            for it in j["items"]:
                if it["number"] in wanted[repo]:
                    found[(repo, it["number"])] = {
                        "repo": repo, "number": it["number"], "title": it["title"], "body": it.get("body") or "",
                        "state": it["state"], "labels": [lb["name"] for lb in it.get("labels", [])],
                        "created_at": it["created_at"][:10], "updated_at": it["updated_at"][:10],
                        "comments": it["comments"], "url": it["html_url"]}
            if len(j["items"]) < 100 or page * 100 >= min(1000, j["total_count"]):
                break
            page += 1
        print(repo, sum(1 for k in found if k[0] == repo), "of", len(wanted[repo]), "issues", flush=True)
    out = [found[(r, n)] for r, n in manifest["issues"]["list"] if (r, n) in found]
    missing = len(manifest["issues"]["list"]) - len(out)
    if missing:
        print(f"warning: {missing} issues of the snapshot were not found (deleted, transferred or private)")
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def issue_passages(issues):
    out = []
    for it in issues:
        body = re.sub(r"```.*?```", "[code omitted]", it["body"], flags=re.S)
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        pieces = split_words(clean_md(body))[:2] or ["(no description)"]
        doc = f"gh-{slug(it['repo'])}-{it['number']}"
        for i, piece in enumerate(pieces, 1):
            out.append({"id": f"{doc}-c{i}", "doc": doc, "index": "github-issues", "doc_type": "github-issue",
                        "status": "unofficial", "title": it["title"], "section": f"Issue #{it['number']}",
                        "text": piece, "repo": it["repo"], "issue_number": it["number"], "state": it["state"],
                        "labels": it["labels"], "comments": it["comments"], "url": it["url"],
                        "path": f"{it['repo']}/issues/{it['number']}", "date": it["created_at"],
                        "updated": it["updated_at"], "chunk": i})
    return out


def samples(manifest, out_dir):
    base = f"https://raw.githubusercontent.com/opensearch-project/OpenSearch-Dashboards/{manifest['dashboards']}/"
    for name, path in SAMPLE.items():
        dest = os.path.join(out_dir, f"sample_{name}.json")
        if os.path.exists(dest):
            continue
        r = requests.get(base + SAMPLE_PATH + path, timeout=300)
        r.raise_for_status()
        lines = gzip.decompress(r.content).decode().splitlines()
        json.dump([json.loads(x) for x in lines[:SAMPLE_DOCS]], open(dest, "w"))
        print("sample", name, len(lines), "->", SAMPLE_DOCS, flush=True)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verify(chunks, manifest):
    by_id = {c["id"]: c for c in chunks}
    questions = json.load(open(os.path.join(DATA, "questions_real.json")))["questions"]
    ids = sorted({i for q in questions for i in q["gold"] + q["ok"]})
    missing = [i for i in ids if i not in by_id]
    changed = [i for i in ids if i in by_id and sha(by_id[i]["text"]) != manifest["gold_sha256"].get(i)]
    counts = collections.Counter(c["index"] for c in chunks)
    print("passages per source:", dict(counts), "expected:", manifest["passages"])
    if missing or changed:
        raise SystemExit(f"The rebuilt corpus does not match the snapshot: {len(missing)} gold passages missing "
                         f"({missing[:5]}), {len(changed)} changed ({changed[:5]}). Check the pinned commits in "
                         "data/real_corpus_manifest.json and that the clones are clean.")
    print(f"all {len(ids)} gold passages of the real questions match the snapshot")


def main():
    cfg = Config()
    manifest = json.load(open(os.path.join(DATA, "real_corpus_manifest.json")))
    src = cfg.wpath("src_real")
    os.makedirs(src, exist_ok=True)
    docs, build = os.path.join(src, "documentation-website"), os.path.join(src, "opensearch-build")
    if "--skip-clone" not in sys.argv:
        clone("https://github.com/opensearch-project/documentation-website.git", docs,
              manifest["documentation_website"], [f for _, fs in DOC_SETS.values() for f in fs])
        clone("https://github.com/opensearch-project/opensearch-build.git", build, manifest["opensearch_build"],
              ["release-notes"])
    samples(manifest, src)
    issues = fetch_issues(manifest, os.path.join(src, "issues.json"))
    chunks = (docs_passages(docs, manifest["documentation_website"])
              + release_passages(build, manifest["opensearch_build"]) + issue_passages(issues))
    for c in chunks:
        for k in ("text", "title", "section"):
            c[k] = c[k].replace("${parameters", "$ {parameters")
    dup = [k for k, v in collections.Counter(c["id"] for c in chunks).items() if v > 1]
    assert not dup, dup[:5]
    json.dump({"chunks": chunks}, open(cfg.wpath("corpus_real.json"), "w"))
    print(len(chunks), "passages written to", cfg.wpath("corpus_real.json"))
    verify(chunks, manifest)


if __name__ == "__main__":
    main()
