# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic corpus (data/corpus_synth.json): passages about Nimbus, a fictional software company, in 17 collections
with collection-specific field names and about 25 fields each. All values are synthetic. Every content document has
status official | unofficial."""

import hashlib
import json
import os

from common import DATA

STATUS = {"pricing-archive": "superseded", "legacy-docs": "retired", "orbit-changelog": "third_party",
          "support-tickets": "unreviewed", "community-forum": "unreviewed", "product-roadmap": "planned",
          "internal-wiki": "draft", "sales-playbook": "internal"}
UNOFFICIAL = set(STATUS)
TEAMS = ["docs", "billing", "platform", "sync", "gateway", "analytics", "security", "support", "marketing", "sales"]

K, T, D, I, B, F = ({"type": "keyword"}, {"type": "text"}, {"type": "date", "format": "yyyy-MM-dd"},
                    {"type": "integer"}, {"type": "boolean"}, {"type": "float"})

# collection -> (field for chunk title/doc title, section heading, text, product, plans, date, extra fields)
LAYOUT = {
    "product-manuals": ("doc_title", "section_heading", "body_text", "product_line", "plans", "last_updated",
                        {"file": K, "audience": K, "product_version": K, "page_no": I, "chunk_no": I}),
    "api-reference": ("title", "section", "content", "product", "plans", "published",
                      {"api_version": K, "url": K, "page": I, "chunk": I}),
    "pricing": ("sheet_title", "section_title", "details", "product", "tier", "effective_date",
                {"sheet": K, "sheet_version": K, "page": I, "chunk": I, "currency": K}),
    "release-notes": ("summary", "heading", "notes", "product", "applies_to", "release_date",
                      {"version": K, "path": K, "chunk": I}),
    "help-center": ("question", "heading", "answer", "product", "plans", "updated_at",
                    {"category": K, "helpful_votes": I, "chunk": I}),
    "security-compliance": ("title", "section", "content", "product", "plans", "published_date",
                            {"framework": K, "file": K, "page": I, "chunk": I}),
    "pricing-archive": ("sheet_title", "section_title", "details", "product", "tier", "effective_date",
                        {"sheet": K, "sheet_version": K, "superseded": B, "page": I, "chunk": I}),
    "legacy-docs": ("doc_title", "section_heading", "body_text", "product_line", "plans", "last_updated",
                    {"product_version": K, "end_of_support": K, "page_no": I, "chunk_no": I}),
    "orbit-changelog": ("title", "heading", "changes", "product", "plans", "released",
                        {"vendor": K, "version": K, "chunk": I}),
    "support-tickets": ("subject", "heading", "conversation", "product", "plan", "opened_at",
                        {"ticket_id": K, "ticket_status": K, "priority": K, "channel": K, "csat": I, "chunk": I}),
    "community-forum": ("thread_title", "heading", "posts", "product", "plans", "posted_at",
                        {"thread_no": I, "replies": I, "views": I, "solved": B, "chunk": I}),
    "product-roadmap": ("title", "heading", "description", "product", "plans", "updated",
                        {"target": K, "item_status": K, "page": I, "chunk": I}),
    "internal-wiki": ("page_title", "heading", "body", "product", "plans", "last_edited",
                      {"space": K, "chunk": I}),
    "sales-playbook": ("title", "heading", "body", "product", "plans", "updated",
                       {"segment": K, "page": I, "chunk": I}),
    "marketing-blog": ("title", "heading", "body", "product", "plans", "published_at",
                       {"author": K, "chunk": I}),
    "training-catalog": ("course_title", "heading", "description", "product", "plans", "updated",
                         {"format": K, "chunk": I}),
    "incident-reports": ("title", "heading", "summary_text", "product", "plans", "reported_at",
                         {"severity": K, "region": K, "chunk": I}),
}
NOISE = {"doc_ref": K, "language": K, "owner_team": K, "status": K, "revision": I, "confidentiality": K,
         "url_path": K, "tags": K, "created_at": D, "indexed_at": D, "checksum": K, "word_count": I,
         "page_count": I, "chunk_count": I, "region_availability": K, "reading_time_min": I}


def chunks():
    return json.load(open(os.path.join(DATA, "corpus_synth.json")))["chunks"]


def _h(s, n=8):
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def mappings():
    out = {}
    for coll, (title, heading, text, product, plans, date, extra) in LAYOUT.items():
        props = {title: T, heading: T, text: T, product: K, plans: K, date: D, **extra, **NOISE}
        out[coll] = {"properties": props}
    return out


def _noise(c):
    words = len(c["text"].split())
    return {"doc_ref": c["doc"], "language": "en", "owner_team": TEAMS[int(_h(c["doc"]), 16) % len(TEAMS)],
            "status": "unofficial" if c["coll"] in UNOFFICIAL else "official", "revision": 1 + int(_h(c["id"]), 16) % 7,
            "confidentiality": "internal" if c["coll"] in ("internal-wiki", "sales-playbook") else "public",
            "url_path": f"/{c['coll']}/{c['doc']}#p{c['page']}", "tags": [c["coll"].split("-")[0], c["product"].split()[-1].lower()],
            "created_at": c["date"], "indexed_at": "2026-11-17", "checksum": _h(c["text"], 12), "word_count": words,
            "page_count": c["page_count"], "chunk_count": c["chunk_count"],
            "region_availability": ["us", "eu", "apac"], "reading_time_min": max(1, words // 200)}


def _extra(c):
    coll = c["coll"]
    v = c.get("version") or c.get("product_version") or c.get("api_version") or ""
    e = {"file": c.get("doc_file") or c.get("sheet_file") or f"{c['doc']}.pdf", "audience": c.get("audience", "all"),
         "product_version": c.get("product_version", v), "page_no": c["page"], "chunk_no": c["chunk"],
         "api_version": c.get("api_version", "v2"), "url": f"https://docs.nimbus.example/{c['doc']}", "page": c["page"],
         "chunk": c["chunk"], "sheet": c.get("sheet_file", f"{c['doc']}.pdf"), "sheet_version": v, "currency": "USD",
         "version": v, "path": f"{coll}/{c['doc']}.md", "category": "billing" if "pric" in c["doc_title"].lower() or
         c["doc"].startswith(("faq-refund", "faq-cancel", "faq-payment", "faq-seats", "faq-nonprofit", "faq-billing",
                               "faq-invoice")) else "account",
         "helpful_votes": int(_h(c["doc"]), 16) % 300, "framework": "SOC2" if c["doc"] == "sec-overview" else "general",
         "superseded": True, "end_of_support": "2026-01-31" if c["doc"] == "legacy-sync-4" else "2026-01-01",
         "vendor": "Orbit Labs", "ticket_id": c.get("ticket_no", ""), "ticket_status": c.get("status", ""),
         "priority": ["low", "medium", "high"][int(_h(c["doc"]), 16) % 3], "channel": ["email", "chat", "web"][int(_h(c["doc"] + "c"), 16) % 3],
         "csat": 1 + int(_h(c["doc"] + "s"), 16) % 5, "thread_no": c.get("thread_no", 0),
         "replies": 1 + int(_h(c["doc"] + "r"), 16) % 9, "views": 50 + int(_h(c["doc"] + "v"), 16) % 2000,
         "solved": int(_h(c["doc"] + "x"), 16) % 2 == 0, "target": "2027-H1", "item_status": "planned",
         "space": "PM" if "pricing" in c["doc"] else "ENG", "segment": "mid-market", "author": "Nimbus team",
         "format": "self-paced", "severity": "sev2", "region": "us"}
    return {k: e[k] for k in LAYOUT[coll][6]}


def content_docs():
    """(collection, _id, source) for every synthetic passage."""
    out = []
    for c in chunks():
        title, heading, text, product, plans, date, _ = LAYOUT[c["coll"]]
        src = {title: c["doc_title"], heading: c["heading"], text: c["text"], product: c["product"],
               plans: c["plans"] if len(c["plans"]) > 1 or plans != "plan" else c["plans"][0], date: c["date"],
               **_extra(c), **_noise(c)}
        src = {k: v for k, v in src.items() if v not in ("", None)}
        out.append((c["coll"], c["id"], src))
    return out
