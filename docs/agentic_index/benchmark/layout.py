# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cluster layout: one realistic many-index cluster used by every setup.

Relevant documents (the synthetic Nimbus corpus and the real OpenSearch corpus) are spread over indices the
way they accumulate in practice: per product, per month or release, per team and per repository, with aliases over the
families. Unrelated indices (orders, invoices, logs, metrics, per-tenant copies, tickets, audit, samples, ...) hold
seeded synthetic records with overlapping vocabulary (error codes, plans, SSO). Two data streams hold gateway and sync
service logs.

Conventions:
- Every index with prose text has `status` = official | unofficial; no other field is named `status`.
- One embedding field per embedded index, `<main text field>_embedding` (all-MiniLM-L6-v2, 384 dims), on every
  text-bearing relevant index and on a share of the unrelated ones (EMBEDDED_UNRELATED).
- About a fifth of the indices carry a mapping `_meta.description` (has_description(), a hash of the name).
- Relevant indices use the english analyzer on text fields; unrelated ones use the default analyzer.
- Nimbus field names vary by source (data_synth.LAYOUT); OpenSearch docs, release notes and issues have their own
  schemas. `component` is filled for every OpenSearch passage (docs and release-note highlights included) and Nimbus
  documents keep their product field, so no filter field is sparse within an index.

layout(work_dir) returns IndexSpec objects; the real corpus and the sample data sets are read from work_dir
(fetch_real_corpus.py writes them)."""

import hashlib
import json
import os
import random
from dataclasses import dataclass, field

import data_synth

K, T, D, I, B, F = ({"type": "keyword"}, {"type": "text"}, {"type": "date", "format": "yyyy-MM-dd"},
                    {"type": "integer"}, {"type": "boolean"}, {"type": "float"})
TS = {"type": "date"}
T_EN = {"type": "text", "analyzer": "english"}
KNN = {"type": "knn_vector", "dimension": 384, "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}


@dataclass
class IndexSpec:
    name: str
    family: str
    purpose: str
    mapping: dict
    docs: list = field(default_factory=list)  # (_id, source)
    embed_field: str = None  # text field whose embedding is stored in <field>_embedding
    relevant: bool = False
    data_stream: bool = False


def has_description(name):
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 5 == 0


def _h(s, mod):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % mod


def _en(mapping):
    """english analyzer on every text field (relevant indices)."""
    props = {}
    for f, d in mapping["properties"].items():
        props[f] = dict(d, analyzer="english") if d.get("type") == "text" else d
    return {"properties": props}


# ---------------- Nimbus (synthetic corpus) ----------------

NIMBUS_PRODUCTS = {"Nimbus Platform": "platform", "Nimbus Sync": "sync", "Nimbus Gateway": "gateway",
                   "Nimbus Analytics": "analytics"}


def _quarter(date):
    return f"{date[:4]}.q{(int(date[5:7]) - 1) // 3 + 1}"


def nimbus_index(coll, c, src):
    """Concrete index of one Nimbus chunk."""
    if coll == "product-manuals":
        return f"docs-nimbus-{NIMBUS_PRODUCTS[c['product']]}"
    if coll == "api-reference":
        return "docs-nimbus-api"
    if coll == "security-compliance":
        return "docs-nimbus-trust-center"
    if coll == "legacy-docs":
        return "docs-nimbus-legacy"
    if coll == "pricing":
        return f"pricing-nimbus-{c['date'][:4]}.{c['date'][5:7]}"
    if coll == "pricing-archive":
        return f"pricing-archive-nimbus-{c['date'][:4]}"
    if coll == "release-notes":
        return f"release-notes-nimbus-{c['date'][:4]}.{c['date'][5:7]}"
    if coll == "help-center":
        return f"support-help-nimbus-{NIMBUS_PRODUCTS[c['product']]}"
    if coll == "support-tickets":
        return f"support-tickets-nimbus-{c['date'][:4]}.{c['date'][5:7]}"
    if coll == "community-forum":
        return "community-forum-nimbus"
    if coll == "product-roadmap":
        return "roadmap-nimbus"
    if coll == "internal-wiki":
        return "wiki-pm" if src.get("space") == "PM" else "wiki-eng"
    if coll == "sales-playbook":
        return "sales-playbook"
    if coll == "marketing-blog":
        return "blog-nimbus"
    if coll == "training-catalog":
        return "training-catalog"
    if coll == "incident-reports":
        return f"incident-reports-{_quarter(c['date'])}"
    if coll == "orbit-changelog":
        return "competitor-orbit-changelog"
    raise KeyError(coll)


NIMBUS_PURPOSE = {
    "docs-nimbus-": "Official Nimbus {p} product manual (admin, user and troubleshooting guide).",
    "docs-nimbus-api": "Official Nimbus Gateway REST API reference.",
    "docs-nimbus-trust-center": "Official Nimbus security, compliance and SLA documents.",
    "docs-nimbus-legacy": "Manuals of retired Nimbus product versions (Sync 4.x, Analytics 1.x).",
    "pricing-nimbus-": "Official Nimbus pricing sheet published {m}.",
    "pricing-archive-nimbus-": "Superseded Nimbus pricing sheet from {y}, kept for reference.",
    "release-notes-nimbus-": "Official Nimbus release notes published in {m}.",
    "support-help-nimbus-": "Official Nimbus {p} help-center articles (billing and account FAQs).",
    "support-tickets-nimbus-": "Nimbus customer support tickets opened in {m}, with agent replies.",
    "community-forum-nimbus": "Nimbus community forum threads written by customers.",
    "roadmap-nimbus": "Planned, unreleased Nimbus features.",
    "wiki-pm": "Product-management wiki drafts and proposals (not approved).",
    "wiki-eng": "Engineering wiki drafts and design notes (not approved).",
    "sales-playbook": "Internal sales guidance and competitive notes.",
    "blog-nimbus": "Nimbus marketing blog posts.",
    "training-catalog": "Nimbus training courses and certifications.",
    "incident-reports-": "Post-incident reports for Nimbus service outages, {q}.",
    "competitor-orbit-changelog": "Changelog of Orbit Sync, a competitor's product.",
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _month_name(ym):
    y, m = ym.split(".")
    return f"{MONTHS[int(m) - 1]} {y}"


def nimbus_purpose(name):
    if name in NIMBUS_PURPOSE:
        return NIMBUS_PURPOSE[name]
    for prefix, text in NIMBUS_PURPOSE.items():
        if name == prefix or (prefix.endswith("-") and name.startswith(prefix)):
            tail = name[len(prefix):] if prefix.endswith("-") else ""
            p = {"platform": "Platform", "sync": "Sync", "gateway": "Gateway", "analytics": "Analytics"}.get(tail, tail)
            return text.format(p=p, m=_month_name(tail) if "." in tail and "q" not in tail else tail,
                               y=tail, q=tail.replace(".q", " Q"))
    raise KeyError(name)


# Dropped from Nimbus documents so that the plugin's 20-field passage metadata cap (PassageStore) keeps the plan, version
# and status fields; the other 12 noise fields stay.
NIMBUS_DROPPED_NOISE = ("checksum", "indexed_at", "reading_time_min", "revision")


def nimbus_specs(chunks_synth):
    mappings = data_synth.mappings()
    for m in mappings.values():
        for f in NIMBUS_DROPPED_NOISE:
            m["properties"].pop(f, None)
    specs = {}
    for coll, _id, src in data_synth.content_docs():
        src = {k: v for k, v in src.items() if k not in NIMBUS_DROPPED_NOISE}
        c = chunks_synth[_id]
        name = nimbus_index(coll, c, src)
        if name not in specs:
            text_field = data_synth.LAYOUT[coll][2]
            specs[name] = IndexSpec(name, "nimbus:" + coll, nimbus_purpose(name), _en(mappings[coll]),
                                    embed_field=text_field, relevant=True)
        specs[name].docs.append((_id, src))
    return list(specs.values())


# ---------------- OpenSearch (real corpus) ----------------

ML_AGENTS = {"Tools", "Agents", "Agents and tools", "Memory and context", "Agentic memory", "Using MCP tools",
             "MCP servers and clients"}
AI_SEARCH_FLOW = {"AI search workflows", "Building AI search workflows in OpenSearch Dashboards", "Workflow APIs",
                  "Workflow template security", "Filling search templates"}
AI_SEARCH_NEURAL = {"Agentic search", "AI search", "Hybrid search", "Neural sparse search"}
RELEVANCE = {"Search Relevance Workbench": "Search Relevance", "Learning to Rank": "Learning to Rank",
             "User Behavior Insights": "User Behavior Insights", "Reranking search results": "Search Relevance",
             "Optimizing search quality": "Search Relevance", "Compare Search Results": "Search Relevance"}


def os_docs_index(c):
    """(index, component) of one OpenSearch documentation passage."""
    p, idx = c["parent"], c["index"]
    if idx == "docs-ml-commons":
        if p.endswith("APIs"):
            return "docs-opensearch-ml-apis", "ML Commons"
        if p in ML_AGENTS:
            return "docs-opensearch-ml-agents", "ML Commons"
        return "docs-opensearch-ml-models", "ML Commons"
    if idx == "docs-vector-search":
        if p in AI_SEARCH_FLOW:
            return "docs-opensearch-ai-search", "Flow Framework"
        if p in AI_SEARCH_NEURAL:
            return "docs-opensearch-ai-search", "Neural Search"
        return "docs-opensearch-vector-search", "k-NN"
    if idx == "docs-search":
        if p in RELEVANCE:
            return "docs-opensearch-search-relevance", RELEVANCE[p]
        return "docs-opensearch-search", "OpenSearch"
    if p == "Document APIs":
        return "docs-opensearch-api-document", "OpenSearch"
    return "docs-opensearch-api-search", "OpenSearch"


OS_PURPOSE = {
    "docs-opensearch-ml-apis": "Official OpenSearch ML Commons API reference (models, agents, memory, connectors).",
    "docs-opensearch-ml-agents": "Official OpenSearch docs on agents, tools, MCP and agent memory.",
    "docs-opensearch-ml-models": "Official OpenSearch docs on connecting, deploying and using ML models.",
    "docs-opensearch-ai-search": "Official OpenSearch docs on AI, hybrid, neural sparse and agentic search and AI "
                                 "search workflows.",
    "docs-opensearch-vector-search": "Official OpenSearch k-NN vector search documentation.",
    "docs-opensearch-search-relevance": "Official OpenSearch docs on search relevance, Learning to Rank, reranking "
                                        "and User Behavior Insights.",
    "docs-opensearch-search": "Official OpenSearch docs on search features, search pipelines, processors and caching.",
    "docs-opensearch-api-document": "Official OpenSearch document API reference.",
    "docs-opensearch-api-search": "Official OpenSearch search API reference.",
}
DOCS_MAPPING = {"properties": {"title": T_EN, "section": T_EN, "body": T_EN, "component": K, "parent": K,
                               "grand_parent": K, "path": K, "url": K, "last_updated": D, "chunk": I,
                               "word_count": I, "status": K}}
RN_MAPPING = {"properties": {"title": T_EN, "section": T_EN, "notes": T_EN, "version": K, "category": K,
                             "component": K, "release_date": D, "source_file": K, "url": K, "chunk": I, "status": K}}
GH_MAPPING = {"properties": {"title": T_EN, "body": T_EN, "repo": K, "issue_number": I, "state": K, "labels": K,
                             "comments": I, "created_at": D, "updated_at": D, "url": K, "chunk": I, "status": K}}


def os_specs(chunks_real):
    specs = {}

    def add(name, purpose, mapping, embed, _id, src):
        if name not in specs:
            specs[name] = IndexSpec(name, "opensearch:" + name.rsplit("-", 1)[0] if name.startswith("release")
                                    else "opensearch:" + name, purpose, mapping, embed_field=embed, relevant=True)
        specs[name].docs.append((_id, {k: v for k, v in src.items() if v not in ("", None, [])}))

    for c in chunks_real.values():
        if c["index"].startswith("docs-"):
            name, comp = os_docs_index(c)
            src = {"title": c["title"], "section": c["section"], "body": c["text"], "component": comp,
                   "parent": c["parent"], "grand_parent": c["grand_parent"], "path": c["path"], "url": c["url"],
                   "last_updated": c["date"], "chunk": c["chunk"], "word_count": len(c["text"].split()),
                   "status": c["status"]}
            add(name, OS_PURPOSE[name], DOCS_MAPPING, "body", c["id"], src)
        elif c["index"] == "release-notes":
            minor = ".".join(c["version"].split(".")[:2])
            name = f"release-notes-opensearch-{minor}"
            src = {"title": c["title"], "section": c["section"], "notes": c["text"], "version": c["version"],
                   "category": c["category"], "component": c["component"] or "OpenSearch", "release_date": c["date"],
                   "source_file": c["path"], "url": c["url"], "chunk": c["chunk"], "status": c["status"]}
            add(name, f"Official OpenSearch {minor}.x release notes.", RN_MAPPING, "notes", c["id"], src)
        else:
            name = f"github-issues-{c['repo'].lower()}"
            src = {"title": c["title"], "body": c["text"], "repo": c["repo"], "issue_number": c["issue_number"],
                   "state": c["state"], "labels": c["labels"], "comments": c["comments"], "created_at": c["date"],
                   "updated_at": c["updated"], "url": c["url"], "chunk": c["chunk"], "status": c["status"]}
            add(name, f"GitHub issues of the OpenSearch {c['repo']} repository (community bug reports and proposals).",
                GH_MAPPING, "body", c["id"], src)
    return list(specs.values())


# ---------------- unrelated indices ----------------

PLANS = ["Free", "Starter", "Pro", "Enterprise"]
PRICES = {"Free": [0.0], "Starter": [7.0, 8.0, 9.0], "Pro": [18.0, 20.0, 22.0, 24.0], "Enterprise": [38.0, 41.0, 45.0]}
TENANTS = ["acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka"]
TEAMS = ["sync", "gateway", "platform", "analytics"]
CODES = [("E4012", "sync-client", "token refresh failed; retrying"),
         ("E4290", "gateway", "rate limit exceeded; returned 429"),
         ("E5003", "gateway", "upstream unavailable, client will retry"),
         ("E4019", "sync-client", "device limit reached"),
         ("E2207", "sync-client", "rejected file name with reserved characters"),
         ("E4011", "gateway", "expired API key"), ("E4013", "sync-client", "account suspended"),
         (None, "analytics", "dashboard refresh completed")]
EMBEDDED_UNRELATED = ("tenant-tickets", "jira", "chat-transcripts")


def _day(rnd, y, m, maxd=28):
    return f"{y}-{m:02d}-{rnd.randint(1, maxd):02d}"


def unrelated_specs(work):
    rnd = random.Random(20261119)
    specs = []

    def spec(name, family, purpose, props, docs, embed=None, ds=False):
        specs.append(IndexSpec(name, family, purpose, {"properties": props}, docs, embed_field=embed, data_stream=ds))

    order_props = {"order_id": K, "customer_id": K, "tenant": K, "plan": K, "seats": I, "unit_price_usd": F,
                   "amount_usd": F, "billing": K, "order_date": D, "order_state": K, "note": T}
    for m in range(1, 12):
        docs = []
        for i in range(40):
            plan = rnd.choice(PLANS)
            unit, seats = rnd.choice(PRICES[plan]), rnd.randint(1, 120)
            docs.append((f"ord-2026{m:02d}-{i:03d}", {
                "order_id": f"ORD-2026{m:02d}-{i:03d}", "customer_id": f"cust-{rnd.randint(300, 420)}",
                "tenant": rnd.choice(TENANTS), "plan": plan, "seats": seats, "unit_price_usd": unit,
                "amount_usd": round(unit * seats, 2), "billing": rnd.choice(["annual", "monthly"]),
                "order_date": _day(rnd, 2026, m), "order_state": rnd.choice(["paid", "paid", "pending", "refunded"]),
                "note": rnd.choice(["renewal at legacy price", "partner discount", "multi-year deal", "upgrade",
                                    "includes audit log add-on", "nonprofit discount", "seat expansion"])}))
        spec(f"orders-2026.{m:02d}", "orders", f"Customer orders placed in {MONTHS[m - 1]} 2026.", order_props, docs)

    inv_props = {"invoice_id": K, "customer_id": K, "tenant": K, "period": K, "plan": K, "seats": I,
                 "unit_price_usd": F, "credit_pct": F, "total_usd": F, "issued_at": D, "memo": T}
    for m in range(1, 12):
        docs = []
        for i in range(30):
            plan = rnd.choice(PLANS[1:])
            unit, seats, credit = rnd.choice(PRICES[plan]), rnd.randint(5, 200), rnd.choice([0, 0, 0, 5, 10])
            docs.append((f"inv-2026{m:02d}-{i:03d}", {
                "invoice_id": f"INV-2026{m:02d}-{i:03d}", "customer_id": f"cust-{rnd.randint(300, 420)}",
                "tenant": rnd.choice(TENANTS), "period": f"2026-{m:02d}", "plan": plan, "seats": seats,
                "unit_price_usd": unit, "credit_pct": credit, "total_usd": round(seats * unit * (1 - credit / 100), 2),
                "issued_at": f"2026-{m:02d}-01", "memo": "SLA credit applied" if credit else "standard invoice"}))
        spec(f"invoices-2026.{m:02d}", "invoices", f"Invoices issued for {MONTHS[m - 1]} 2026.", inv_props, docs)

    log_props = {"@timestamp": TS, "level": K, "service": K, "host": K, "error_code": K, "workspace_id": K, "plan": K,
                 "message": T, "trace_id": K}
    for d in range(8, 18):
        docs = []
        for i in range(60):
            code, svc, msg = rnd.choice(CODES)
            doc = {"@timestamp": f"2026-11-{d:02d}T{rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}:00Z",
                   "level": "INFO" if code is None else rnd.choice(["WARN", "ERROR"]), "service": svc,
                   "host": f"ip-10-0-{rnd.randint(1, 40)}-{rnd.randint(2, 250)}",
                   "workspace_id": f"ws-{rnd.randint(100, 160)}", "plan": rnd.choice(PLANS), "message": msg,
                   "trace_id": f"{rnd.getrandbits(48):012x}"}
            if code:
                doc["error_code"] = code
            docs.append((f"log-1126{d:02d}-{i:03d}", doc))
        spec(f"app-logs-2026.11.{d:02d}", "app-logs", f"Application logs of all Nimbus services, 2026-11-{d:02d}.",
             log_props, docs)

    met_props = {"@timestamp": TS, "service": K, "region": K, "metric": K, "value": {"type": "double"}, "unit": K}
    metrics = [("gateway", "availability_pct", "%", 99.93), ("gateway", "p99_latency_ms", "ms", 182.0),
               ("gateway", "burst_rejections", "count", 412.0), ("gateway", "requests_per_sec", "rps", 1830.0)]
    for d in range(8, 18):
        docs = []
        for i in range(40):
            svc, met, unit, v = metrics[i % len(metrics)]
            docs.append((f"met-1126{d:02d}-{i:03d}", {
                "@timestamp": f"2026-11-{d:02d}T{i % 24:02d}:00:00Z", "service": svc,
                "region": ["us", "eu", "apac"][i % 3], "metric": met, "unit": unit,
                "value": round(v * rnd.uniform(0.97, 1.03), 3)}))
        spec(f"metrics-gateway-2026.11.{d:02d}", "metrics", f"Gateway service metrics, 2026-11-{d:02d}.", met_props,
             docs)

    ds_props = {"@timestamp": TS, "level": K, "host": K, "error_code": K, "workspace_id": K, "message": T,
                "latency_ms": I}
    for svc, codes in (("gateway", ["E4290", "E5003", "E4011", None]), ("sync", ["E4012", "E4019", "E2207", None])):
        docs = []
        for i in range(80):
            code = rnd.choice(codes)
            msg = next((m for c, _, m in CODES if c == code), "request completed") if code else "request completed"
            doc = {"@timestamp": f"2026-11-{rnd.randint(1, 18):02d}T{rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}:00Z",
                   "level": "INFO" if code is None else "ERROR", "host": f"{svc}-{rnd.randint(1, 12)}",
                   "workspace_id": f"ws-{rnd.randint(100, 160)}", "message": msg, "latency_ms": rnd.randint(5, 900)}
            if code:
                doc["error_code"] = code
            docs.append((f"ds-{svc}-{i:03d}", doc))
        spec(f"logs-nimbus-{svc}-prod", "data-stream", f"Data stream of production Nimbus {svc} service logs.",
             ds_props, docs, ds=True)

    user_props = {"user_id": K, "email": K, "role": K, "plan": K, "sso_enabled": B, "idp": K, "devices": I,
                  "created_at": D, "last_login": TS}
    tkt_props = {"ticket_id": K, "opened_at": D, "priority": K, "ticket_state": K, "product": K, "subject": T,
                 "description": T, "status": K}
    usage_props = {"date": D, "active_users": I, "api_calls": I, "storage_gb": F, "sync_conflicts": I}
    issues = [
        ("Sync stops with E4012 after laptop wakes", "Several users see E4012 in Nimbus Sync each morning after the "
         "laptop wakes from sleep; signing in again fixes it until the next day.", "Nimbus Sync"),
        ("429 errors from the API during nightly export", "Our nightly export job gets HTTP 429 with E4290 after "
         "about two minutes of calls; we think we hit the gateway rate limit.", "Nimbus Gateway"),
        ("New laptop cannot sync, device limit", "A user replaced a laptop and the new one shows E4019; the old "
         "device still counts against the limit.", "Nimbus Sync"),
        ("Question about SSO with our IdP", "We want to enforce SSO for every member and asked which plan we need; "
         "we currently sign in with passwords.", "Nimbus Platform"),
        ("Dashboard shows stale numbers", "Analytics dashboards look a few minutes behind the source data; is the "
         "refresh interval configurable?", "Nimbus Analytics"),
        ("Files with colons are skipped", "Files whose names contain ':' are rejected with E2207 and do not sync.",
         "Nimbus Sync"),
        ("Invoice shows the wrong seat count", "Our latest invoice lists more seats than we have active members.",
         "Nimbus Platform"),
        ("Gateway returned 503 for a few minutes", "Calls failed with E5003 for several minutes this morning; "
         "the status page showed nothing.", "Nimbus Gateway"),
        ("API key stopped working", "Requests fail with E4011 although nobody rotated the key.", "Nimbus Gateway"),
        ("Guests cannot open shared folders", "External guests get an access error on a shared folder.",
         "Nimbus Platform"),
    ]
    idps = ["okta", "entra-id", "google-workspace", "onelogin", "ping-identity"]
    for t in TENANTS:
        trnd = random.Random("tenant-" + t)
        plan = trnd.choice(PLANS[1:])
        docs = []
        for i in range(25):
            sso = plan != "Starter" and trnd.random() < 0.7
            d = {"user_id": f"{t}-u{i + 1:03d}", "email": f"user{i + 1}@{t}.example", "plan": plan,
                 "role": trnd.choice(["Owner", "Admin", "Member", "Member", "Guest"]), "sso_enabled": sso,
                 "devices": trnd.randint(1, 9), "created_at": _day(trnd, 2026, trnd.randint(1, 10)),
                 "last_login": f"2026-11-{trnd.randint(1, 18):02d}T09:00:00Z"}
            if sso:
                d["idp"] = trnd.choice(idps)
            docs.append((f"{t}-u{i + 1:03d}", d))
        spec(f"tenant-{t}-users", "tenant-users", f"Users of the {t} tenant workspace.", user_props, docs)
        docs = []
        for i in range(15):
            subj, desc, prod = issues[trnd.randrange(len(issues))]
            n = trnd.randint(2, 40)
            desc2 = (f"{desc} Tenant {t.capitalize()} ({plan} plan): about {n} users affected; "
                     f"first seen {MONTHS[trnd.randint(7, 10)]} {trnd.randint(1, 28)}, 2026.")
            docs.append((f"{t}-t{i + 1:03d}", {
                "ticket_id": f"{t.upper()}-{100 + i}", "opened_at": _day(trnd, 2026, trnd.randint(6, 11)),
                "priority": trnd.choice(["low", "medium", "high"]),
                "ticket_state": trnd.choice(["open", "pending", "solved", "solved"]), "product": prod,
                "subject": subj, "description": desc2, "status": "unofficial"}))
        spec(f"tenant-{t}-tickets", "tenant-tickets", f"Support tickets raised by the {t} tenant.", tkt_props, docs,
             embed="description")
        docs = []
        for i in range(30):
            docs.append((f"{t}-us{i + 1:03d}", {"date": f"2026-10-{i + 1:02d}" if i < 31 else "2026-10-31",
                                                "active_users": trnd.randint(10, 400),
                                                "api_calls": trnd.randint(1000, 90000),
                                                "storage_gb": round(trnd.uniform(5, 900), 1),
                                                "sync_conflicts": trnd.randint(0, 30)}))
        spec(f"tenant-{t}-usage", "tenant-usage", f"Daily usage of the {t} tenant, October 2026.", usage_props, docs)

    for m in (9, 10, 11):
        docs = []
        actions = ["sso.enabled", "user.deleted", "legal_hold.created", "share.public_link", "audit.export",
                   "2fa.required", "api_key.rotated"]
        for i in range(50):
            docs.append((f"ae-2026{m:02d}-{i:03d}", {
                "@timestamp": f"2026-{m:02d}-{rnd.randint(1, 28):02d}T12:00:00Z", "tenant": rnd.choice(TENANTS),
                "actor": f"u-{rnd.randint(1, 40):04d}", "action": rnd.choice(actions), "target": f"obj-{i}",
                "ip": f"203.0.113.{i}", "details": "admin action from the console"}))
        spec(f"audit-events-2026.{m:02d}", "audit", f"Workspace admin audit events, {MONTHS[m - 1]} 2026.",
             {"@timestamp": TS, "tenant": K, "actor": K, "action": K, "target": K, "ip": K, "details": T}, docs)

    help_topics = [("VPN drops every hour", "Corporate VPN disconnects about every hour; reconnecting works."),
                   ("Okta MFA push not arriving", "Okta Verify pushes are not delivered on my new phone."),
                   ("Laptop fan loud during builds", "My laptop fan runs at full speed during builds."),
                   ("Need access to the Nimbus admin console", "Please grant me admin access to test SSO."),
                   ("Test workspace shows E4012", "Our QA workspace shows E4012 after the token lifetime change."),
                   ("Printer on floor 3 offline", "The floor 3 printer shows offline."),
                   ("Request a second monitor", "I would like a second monitor for my desk.")]
    for q in ("2026.q3", "2026.q4"):
        docs = []
        for i in range(25):
            s, b = rnd.choice(help_topics)
            docs.append((f"it-{q}-{i:03d}", {"ticket_id": f"IT-{q[-2:].upper()}-{i:03d}", "requester": f"emp-{rnd.randint(1, 300)}",
                                             "opened_at": _day(rnd, 2026, rnd.randint(7, 9) if q.endswith("3") else rnd.randint(10, 11)),
                                             "category": rnd.choice(["network", "identity", "hardware", "access"]),
                                             "subject": s, "body": b, "it_state": rnd.choice(["open", "closed"]),
                                             "status": "unofficial"}))
        spec(f"helpdesk-it-{q}", "helpdesk", f"Internal IT helpdesk tickets, {q.replace('.q', ' Q')}.",
             {"ticket_id": K, "requester": K, "opened_at": D, "category": K, "subject": T, "body": T, "it_state": K,
              "status": K}, docs)

    jira = {"sync": [("Investigate E4012 after sleep on macOS", "Token refresh races the network coming back after "
                      "sleep; proposal: retry refresh with backoff before showing E4012.", "bug"),
                     ("Raise the device limit for Enterprise", "Proposal to raise the Enterprise device limit; not "
                      "decided.", "story"),
                     ("E2207 on files with trailing spaces", "Reserved-character check also rejects trailing spaces.",
                      "bug")],
            "gateway": [("Proposal: Pro burst limit 150 rps", "Product asked to evaluate raising the Pro burst limit to "
                         "150 requests per second. Experiment behind flag burst_150; nothing shipped.", "story"),
                        ("E5003 spikes during upstream deploys", "Upstream deploys cause short E5003 spikes.", "bug"),
                        ("Return Retry-After on E4290", "Add a Retry-After header to 429 responses.", "story")],
            "platform": [("SCIM for Pro (spike)", "Spike to estimate SCIM provisioning on the Pro plan; no decision.",
                          "spike"),
                         ("Seat count off by one after downgrade", "Invoices count a removed seat for one cycle.",
                          "bug"),
                         ("Passkeys as primary login", "Passkey login for Enterprise, rollout behind a flag.", "story")],
            "analytics": [("Configurable refresh interval", "Customers ask for a refresh interval below one minute.",
                           "story"),
                          ("Dashboard export times out", "CSV export of large dashboards times out.", "bug"),
                          ("Free plan analytics experiment", "Experiment: limited analytics on Free.", "spike")]}
    for team, items in jira.items():
        docs = []
        for i in range(25):
            s, d, typ = items[i % len(items)]
            docs.append((f"jira-{team}-{i:03d}", {"key": f"{team[:3].upper()}-{1000 + i}", "summary": s,
                                                  "description": d + f" (comment {i})", "issue_type": typ,
                                                  "priority": rnd.choice(["P1", "P2", "P3"]),
                                                  "jira_state": rnd.choice(["To Do", "In Progress", "Done"]),
                                                  "assignee": f"eng-{rnd.randint(1, 60)}",
                                                  "created": _day(rnd, 2026, rnd.randint(5, 11)),
                                                  "labels": [team, typ], "status": "unofficial"}))
        spec(f"jira-{team}", "jira", f"Engineering issue tracker of the {team} team.",
             {"key": K, "summary": T, "description": T, "issue_type": K, "priority": K, "jira_state": K,
              "assignee": K, "created": D, "labels": K, "status": K}, docs, embed="description")

    queries = ["E4012", "pro plan price", "sso okta", "device limit", "cancel subscription", "E4290 rate limit",
               "data residency tokyo", "refund policy", "audit log retention", "scim", "invoice download",
               "sync 5.0 release", "api key rotate", "guest access", "nonprofit discount"]
    for m in (10, 11):
        docs = [(f"sq-2026{m:02d}-{i:03d}", {"@timestamp": f"2026-{m:02d}-{rnd.randint(1, 28):02d}T10:00:00Z",
                                             "query": rnd.choice(queries), "results_count": rnd.randint(0, 40),
                                             "clicked_doc": rnd.choice(["", "faq-refund", "pricing-2026-09",
                                                                        "man-sync-p3"]),
                                             "session_id": f"s-{rnd.getrandbits(32):08x}", "status": "unofficial"})
                for i in range(60)]
        spec(f"search-analytics-2026.{m:02d}", "search-analytics",
             f"Help-center search queries typed by users, {MONTHS[m - 1]} 2026.",
             {"@timestamp": TS, "query": T, "results_count": I, "clicked_doc": K, "session_id": K, "status": K}, docs)

    chats = ["Prospect asked whether Pro includes SSO; rep said yes and offered a 10% first-year discount.",
             "Customer on legacy $20 Pro pricing asked when the renewal price changes; rep promised to check.",
             "Prospect compared Nimbus with Orbit Sync; rep sent the comparison sheet.",
             "Enterprise prospect asked for EU data residency; rep confirmed Frankfurt.",
             "Startup asked about nonprofit discounts; rep explained eligibility rules are on the help center."]
    docs = [(f"chat-{i:03d}", {"started_at": _day(rnd, 2026, rnd.randint(10, 11)), "rep": f"rep-{rnd.randint(1, 20)}",
                               "segment": rnd.choice(["smb", "mid-market", "enterprise"]),
                               "transcript": rnd.choice(chats), "status": "unofficial"}) for i in range(20)]
    spec("chat-transcripts-sales-2026.q4", "chat-transcripts", "Sales chat transcripts, 2026 Q4.",
         {"started_at": D, "rep": K, "segment": K, "transcript": T, "status": K}, docs, embed="transcript")

    comments = ["Sync is fast now", "Price went up again", "SSO setup was painless", "Too many E4012 errors",
                "Love the analytics dashboards", "API limits are too low for us", "Support was quick"]
    docs = [(f"nps-{i:03d}", {"submitted_at": _day(rnd, 2026, rnd.randint(1, 11)), "score": rnd.randint(0, 10),
                              "plan": rnd.choice(PLANS), "comment": rnd.choice(comments), "status": "unofficial"})
            for i in range(50)]
    spec("survey-responses-2026", "survey", "NPS survey responses with free-text comments, 2026.",
         {"submitted_at": D, "score": I, "plan": K, "comment": T, "status": K}, docs)

    camp = [("Sync 5 is here", "Faster first sync and smarter conflict handling. Update today."),
            ("Webinar: SSO in 20 minutes", "Join our live session on setting up SSO."),
            ("Your workspace recap", "See what your team shared this month.")]
    docs = [(f"mail-{i:03d}", {"subject": camp[i % 3][0], "body": camp[i % 3][1], "sent_at": _day(rnd, 2026, 6 + i % 6),
                               "segment": rnd.choice(["all", "admins", "trial"]), "open_rate": round(rnd.uniform(.1, .5), 3),
                               "status": "official"}) for i in range(12)]
    spec("email-campaigns", "email", "Marketing email campaigns sent to customers.",
         {"subject": T, "body": T, "sent_at": D, "segment": K, "open_rate": F, "status": K}, docs)

    merch = [("Nimbus hoodie", "Grey hoodie with the Nimbus logo.", 49.0), ("Laptop sticker pack", "Five stickers.", 6.0),
             ("Insulated bottle", "750 ml steel bottle.", 25.0), ("Desk mat", "Large desk mat, 90 x 40 cm.", 32.0)]
    docs = [(f"sku-{i:03d}", {"sku": f"MER-{i:03d}", "name": merch[i % 4][0], "description": merch[i % 4][1],
                              "price_usd": merch[i % 4][2], "in_stock": rnd.random() < .8, "status": "official"})
            for i in range(16)]
    spec("merch-store-products", "merch", "Company merchandise store catalog.",
         {"sku": K, "name": K, "description": T, "price_usd": F, "in_stock": B, "status": K}, docs)

    docs = [(f"ct-{i:03d}", {"eventTime": f"2026-11-{rnd.randint(1, 18):02d}T{rnd.randint(0, 23):02d}:00:00Z",
                             "eventName": rnd.choice(["AssumeRole", "PutObject", "GetObject", "CreateUser"]),
                             "userName": f"svc-{rnd.randint(1, 9)}", "sourceIPAddress": f"198.51.100.{i % 250}",
                             "awsRegion": rnd.choice(["us-west-2", "eu-central-1", "ap-northeast-1"])}) for i in range(60)]
    spec("cloudtrail-2026.11", "cloudtrail", "AWS CloudTrail events of the hosting accounts, November 2026.",
         {"eventTime": TS, "eventName": K, "userName": K, "sourceIPAddress": K, "awsRegion": K}, docs)

    rules = ["Impossible travel login", "Many failed logins", "Public link created on sensitive folder",
             "API key used from new country"]
    docs = [(f"sa-{i:03d}", {"@timestamp": f"2026-11-{rnd.randint(1, 18):02d}T08:00:00Z", "rule": rnd.choice(rules),
                             "severity": rnd.choice(["low", "medium", "high"]), "tenant": rnd.choice(TENANTS),
                             "summary": "Alert raised by the detection pipeline for review.", "status": "unofficial"})
            for i in range(40)]
    spec("security-alerts-2026.11", "security-alerts", "Security detection alerts, November 2026.",
         {"@timestamp": TS, "rule": K, "severity": K, "tenant": K, "summary": T, "status": K}, docs)

    docs = [(f"dep-{i:03d}", {"deployed_at": f"2026-{rnd.randint(9, 11):02d}-{rnd.randint(1, 28):02d}T15:00:00Z",
                              "service": rnd.choice(["sync-api", "gateway", "billing", "analytics-worker"]),
                              "build": f"b{2000 + i}", "env": rnd.choice(["beta", "prod"]),
                              "changelog": rnd.choice(["bump dependencies", "config change", "fix flaky retry",
                                                       "enable feature flag rollout", "revert previous build"]),
                              "status": "unofficial"}) for i in range(40)]
    spec("deploy-events", "deploy", "Deployment events of internal services.",
         {"deployed_at": TS, "service": K, "build": K, "env": K, "changelog": T, "status": K}, docs)

    docs = [(f"slow-{i:03d}", {"@timestamp": f"2026-11-{rnd.randint(1, 18):02d}T11:00:00Z",
                               "index_name": rnd.choice(["orders-2026.10", "app-logs-2026.11.12", "tenant-acme-tickets"]),
                               "query_text": rnd.choice(['{"match":{"message":"E4012"}}', '{"range":{"order_date":{}}}',
                                                         '{"term":{"plan":"Pro"}}']),
                               "took_ms": rnd.randint(500, 9000)}) for i in range(30)]
    spec("slow-queries", "slow-queries", "Slow search query log of this cluster.",
         {"@timestamp": TS, "index_name": K, "query_text": T, "took_ms": I}, docs)

    docs = [(f"emp-{i:03d}", {"employee_id": f"emp-{i:03d}", "team": rnd.choice(["docs", "billing", "platform", "sync",
                                                                                  "gateway", "support", "sales"]),
                              "title": rnd.choice(["Engineer", "Manager", "Writer", "Agent"]),
                              "location": rnd.choice(["Seattle", "Berlin", "Tokyo"]), "start_date": _day(rnd, 2025, 6)})
            for i in range(40)]
    spec("hr-employees", "hr", "Employee directory.",
         {"employee_id": K, "team": K, "title": K, "location": K, "start_date": D}, docs)

    crm_props = {"account_id": K, "name": K, "plan": K, "seats": I, "arr_usd": F, "renewal_date": D, "region": K,
                 "notes": T}
    docs = []
    for i in range(30):
        plan = rnd.choice(PLANS[1:])
        seats = rnd.randint(10, 900)
        docs.append((f"acct-{i + 1:03d}", {"account_id": f"acct-{i + 1:03d}", "name": f"Customer {i + 1}", "plan": plan,
                                           "seats": seats, "arr_usd": round(seats * rnd.choice(PRICES[plan]) * 12, 2),
                                           "renewal_date": f"2027-{1 + i % 12:02d}-15",
                                           "region": ["us", "eu", "apac"][i % 3],
                                           "notes": rnd.choice(["asked about SLA credits", "evaluating Orbit",
                                                                "wants SCIM on Pro", "legacy $20 Pro pricing"])}))
    spec("crm-accounts", "crm", "CRM customer accounts.", crm_props, docs)

    flags = [("passkey_primary_login", "Passkeys as primary login", ["Enterprise"], 5),
             ("pro_scim", "SCIM for Pro", ["Pro"], 0), ("free_analytics", "Analytics on Free", ["Free"], 0),
             ("burst_150", "Pro burst limit 150 rps", ["Pro"], 10), ("sync6_preview", "Sync 6 preview",
                                                                       ["Enterprise"], 2),
             ("tokyo_region", "APAC Tokyo region", ["Enterprise"], 100), ("guest_starter", "Guests on Starter",
                                                                           ["Starter"], 100)]
    docs = [(f, {"flag": f, "description": d, "enabled_plans": e, "rollout_pct": p, "updated_at": f"2026-11-0{i + 1}"})
            for i, (f, d, e, p) in enumerate(flags)]
    spec("feature-flags", "flags", "Feature flag configuration.",
         {"flag": K, "description": T, "enabled_plans": K, "rollout_pct": I, "updated_at": D}, docs)

    for name, f, purpose in (("sample-web-logs", "logs", "OpenSearch Dashboards sample web logs."),
                             ("sample-flights", "flights", "OpenSearch Dashboards sample flight data."),
                             ("sample-ecommerce", "ecommerce", "OpenSearch Dashboards sample ecommerce orders.")):
        raw = json.load(open(os.path.join(work, "src_real", f"sample_{f}.json")))
        docs = [(f"{f}-{i:04d}", {k: v for k, v in d.items() if k != "index"}) for i, d in enumerate(raw, 1)]
        specs.append(IndexSpec(name, "samples", purpose, None, docs))
    return specs


ALIASES = {"release-notes-nimbus": "release-notes-nimbus-*", "release-notes-opensearch": "release-notes-opensearch-*",
           "release-notes": "release-notes-*", "pricing-nimbus": "pricing-nimbus-*", "orders": "orders-*",
           "invoices": "invoices-*", "app-logs": "app-logs-*"}
SCOPE = ["release-notes*", "docs-*", "pricing*", "support-*"]
CLUSTER_LINE = ("an OpenSearch cluster of a software company with about {n} indices: documentation, release notes, "
                "pricing and support content for its Nimbus products and for OpenSearch, plus operational data "
                "(orders, invoices, logs, metrics, users, tickets, audit events)")
_CACHE = {}


def chunks_synth():
    if "synth" not in _CACHE:
        _CACHE["synth"] = {c["id"]: c for c in data_synth.chunks()}
    return _CACHE["synth"]


def chunks_real(work):
    if "real" not in _CACHE:
        path = os.path.join(work, "corpus_real.json")
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; run fetch_real_corpus.py first.")
        _CACHE["real"] = {c["id"]: c for c in json.load(open(path))["chunks"]}
    return _CACHE["real"]


def layout(work):
    if "layout" in _CACHE:
        return _CACHE["layout"]
    specs = nimbus_specs(chunks_synth()) + os_specs(chunks_real(work)) + unrelated_specs(work)
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), "duplicate index names"
    ids = [i for s in specs for i, _ in s.docs]
    assert len(ids) == len(set(ids)), "duplicate document ids"
    _CACHE["layout"] = specs
    return specs


def doc_text(src):
    """Plain text of a record for the judge."""
    return "; ".join(f"{k}: {v}" for k, v in src.items())
