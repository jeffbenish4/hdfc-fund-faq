"""RAG ingestion. Usage: python pipeline.py [scrape|load|chunk|embed|store|all]
Each phase writes its output to disk so every stage can be shown in the demo."""
import json, re, sys, datetime, yaml, requests
from pathlib import Path
from bs4 import BeautifulSoup

CFG = yaml.safe_load(open("config.yaml"))
RAW, PROC, CH, EMB = (Path(f"data/{d}") for d in ("raw", "processed", "chunks", "embeddings"))
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
KEYS = re.compile(r"expense|exit_?load|min_?(sip|lump|invest)|lock_?in|benchmark|risk|riskometer|"
                  r"fund_?(name|house|manager|size|category)|aum|scheme|category|plan|launch|stamp|tax|"
                  r"nav|rating|description|objective|sub_?category", re.I)

def scrape():  # ONLY the 5 URLs - no internal links are followed
    for s in CFG["sources"]:
        r = requests.get(s["url"], headers=UA, timeout=30); r.raise_for_status()
        (RAW / f"{s['id']}.html").write_text(r.text, encoding="utf-8")
        print("scraped", s["id"], len(r.text))

def flatten(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flatten(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o[:40]): yield from flatten(v, f"{path}[{i}]")
    elif o not in (None, "", [], {}): yield path, o

def load():  # raw html -> clean text + structured facts (from the page's embedded JSON)
    for s in CFG["sources"]:
        soup = BeautifulSoup((RAW / f"{s['id']}.html").read_text(encoding="utf-8"), "html.parser")
        facts = []
        nd = soup.find("script", id="__NEXT_DATA__")
        if nd and nd.string:
            for p, v in flatten(json.loads(nd.string).get("props", {})):
                leaf = p.split(".")[-1]
                if KEYS.search(leaf) and len(str(v)) < 300:
                    facts.append(f"{leaf}: {v}")
        for t in soup(["script", "style", "noscript", "svg"]): t.decompose()
        lines = [l.strip() for l in soup.get_text("\n").split("\n") if len(l.strip()) > 2]
        (PROC / f"{s['id']}.json").write_text(json.dumps(
            {**s, "scraped_at": str(datetime.date.today()), "facts": list(dict.fromkeys(facts)), "lines": lines},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("loaded", s["id"], len(facts), "facts", len(lines), "lines")

def chunk():
    # Strategy: (1) structured facts grouped 8 per chunk (precise answers: expense ratio, exit load, SIP...),
    # (2) visible text packed into ~chunk_size char windows with overlap. Every chunk is prefixed with fund name.
    size, ov, out = CFG["chunk_size"], CFG["chunk_overlap"], []
    for s in CFG["sources"]:
        d = json.loads((PROC / f"{s['id']}.json").read_text(encoding="utf-8"))
        def add(text, kind):
            out.append({"id": f"{s['id']}-{len(out)}", "text": f"[{d['fund']}] {text}",
                        "metadata": {"source_url": d["url"], "fund": d["fund"], "type": kind, "scraped_at": d["scraped_at"]}})
        for i in range(0, len(d["facts"]), 8): add("Facts: " + "; ".join(d["facts"][i:i+8]), "facts")
        buf = ""
        for l in d["lines"]:
            if len(buf) + len(l) > size and buf:
                add(buf, "text"); buf = buf[-ov:] + " "
            buf += l + " "
        if buf.strip(): add(buf, "text")
    (CH / "chunks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("chunks:", len(out))

def embed():
    from sentence_transformers import SentenceTransformer
    chunks = json.load(open(CH / "chunks.json", encoding="utf-8"))
    vecs = SentenceTransformer(CFG["embedding_model"]).encode([c["text"] for c in chunks], normalize_embeddings=True)
    json.dump([{"id": c["id"], "embedding": [round(float(x), 4) for x in v]} for c, v in zip(chunks, vecs)],
              open(EMB / "embeddings.json", "w"))
    print("embedded", len(chunks))

def store():
    import chromadb
    chunks = json.load(open(CH / "chunks.json", encoding="utf-8"))
    embs = {e["id"]: e["embedding"] for e in json.load(open(EMB / "embeddings.json"))}
    cl = chromadb.PersistentClient(path=CFG["chroma_path"])
    try: cl.delete_collection(CFG["collection"])
    except Exception: pass
    col = cl.create_collection(CFG["collection"], metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks], embeddings=[embs[c["id"]] for c in chunks])
    print("chroma count:", col.count(), "of", len(chunks), "->", "ALL PRESENT" if col.count() == len(chunks) else "MISSING")
    # export a portable copy for the Vercel runtime
    json.dump({"updated": str(datetime.date.today()), "model": CFG["embedding_model"],
               "chunks": [{**c, "embedding": embs[c["id"]]} for c in chunks]},
              open(CFG["export_path"], "w", encoding="utf-8"), ensure_ascii=False)
    print("exported", CFG["export_path"])

PH = dict(scrape=scrape, load=load, chunk=chunk, embed=embed, store=store)
if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (PH if a == "all" else [a]): PH[n]()
