"""Vercel serverless function: guardrails -> retrieval (cosine) -> Groq LLM -> cited answer."""
import json, os, re, math, urllib.request
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = [os.path.join(HERE, "..", "data", "store.json"), "data/store.json"]
_STORE = None
EDU = "https://www.amfiindia.com/investor-corner/knowledge-center"
PII = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b|\b\d{4}\s?\d{4}\s?\d{4}\b|[\w.+-]+@[\w-]+\.\w+|\b(?:\+?91)?[6-9]\d{9}\b|\botp\b", re.I)
ADVICE = re.compile(r"\b(should i|shall i|better (fund|to)|which (fund|one) is best|best fund|recommend|suggest|buy|sell|invest in|worth (it|investing)|good (fund|investment)|my portfolio|switch)\b", re.I)
PERF = re.compile(r"\b(returns?|cagr|performance|outperform|beat|xirr|profit|compare)\b", re.I)

def store():
    global _STORE
    if _STORE is None:
        for p in PATHS:
            if os.path.exists(p): _STORE = json.load(open(p, encoding="utf-8")); break
    return _STORE

def post(url, body, headers):
       req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", **headers})
    return json.loads(urllib.request.urlopen(req, timeout=40).read())

def embed(q):  # same model as ingestion, via HF Inference API
    r = post("https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
             {"inputs": q}, {"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
    if r and isinstance(r[0], list):  # token-level -> mean pool
        r = [sum(c) / len(r) for c in zip(*r)]
    n = math.sqrt(sum(x * x for x in r)) or 1
    return [x / n for x in r]

def keyword_scores(q, chunks):
    w = set(re.findall(r"[a-z]{3,}", q.lower()))
    return [len(w & set(re.findall(r"[a-z]{3,}", c["text"].lower()))) for c in chunks]

def retrieve(q, k=4):
    ch = store()["chunks"]
    try:
        v = embed(q)
        sc = [sum(a * b for a, b in zip(v, c["embedding"])) for c in ch]
    except Exception:  # fallback if embedding API unavailable
        sc = keyword_scores(q, ch)
    return [ch[i] for i in sorted(range(len(ch)), key=lambda i: -sc[i])[:k]]

def llm(q, ctx):
    sysmsg = ("You are a facts-only mutual fund FAQ assistant. Answer ONLY from the CONTEXT. Max 3 sentences. "
              "No investment advice, no opinions, no return calculations or comparisons. "
              "If the answer is not in the context, say you could not find it in the official pages. Do not include links.")
    r = post("https://api.groq.com/openai/v1/chat/completions",
             {"model": os.environ["GROQ_MODEL"], "temperature": 0, "max_tokens": 400,
              "messages": [{"role": "system", "content": sysmsg},
                           {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"}]},
             {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]})
    t = r["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()

def answer(q):
    q = (q or "").strip()[:500]
    if not q: return {"answer": "Please type a question.", "source": None}
    if PII.search(q):
        return {"answer": "Please don't share personal details like PAN, Aadhaar, account numbers, OTPs, emails or phone numbers. I can only answer general fund facts.", "source": EDU, "refused": True}
    if ADVICE.search(q):
        return {"answer": "Sorry, I can only share facts from official pages and can't give investment advice or buy/sell opinions. This link explains how to evaluate funds yourself.", "source": EDU, "refused": True}
    hits = retrieve(q, int(os.environ.get("TOP_K", 4)))
    if PERF.search(q):
        return {"answer": "I don't compute or compare returns or performance. Please see the scheme's official page and factsheet for the latest figures.", "source": hits[0]["metadata"]["source_url"], "refused": True, "updated": store()["updated"]}
    ctx = "\n---\n".join(h["text"] for h in hits)
    return {"answer": llm(q, ctx), "source": hits[0]["metadata"]["source_url"], "updated": store()["updated"]}

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            out, code = answer(json.loads(self.rfile.read(n)).get("question", "")), 200
        except Exception as e:
            out, code = {"answer": "Something went wrong. Please try again.", "source": None, "error": str(e)[:120]}, 500
        b = json.dumps(out).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
