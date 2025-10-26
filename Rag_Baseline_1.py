# Requires:
#   pip install pypdf faiss-cpu tenacity llama-cpp-python
#   (optional, recommended) pip install sentence-transformers
#
# Tip: If sentence-transformers isn't installed, the script can use a GGUF
#      embedding model (e.g., bge-small-en-v1.5-q4_0.gguf) for 100% local.
#
# Tested with: mistral-7b-instruct-v0.2.Q4_K_M.gguf (great default)
# link for manual model download >> https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

import os, re, json, argparse, logging, math, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from pypdf import PdfReader
import faiss

# Try sentence-transformers
_HAS_ST = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _HAS_ST = False

# llama.cpp for LLM and (optionally) embeddings
from llama_cpp import Llama

import os
from pathlib import Path

def ensure_model_downloaded(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="models"
):
    """
    Downloads the GGUF model from Hugging Face if it's not present locally.
    Requires `pip install huggingface_hub`.
    """
    from huggingface_hub import hf_hub_download

    local_path = Path(local_dir) / filename
    if local_path.exists():
        print(f"[OK] Model već postoji: {local_path}")
        return str(local_path)

    print(f"[*] Preuzimam model {filename} iz {repo_id} ...")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Hugging Face preuzimanje
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[OK] Model preuzet i sačuvan u: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[ERR] Nije moguće preuzeti model: {e}")
        return None


# Automatski proveri i preuzmi ako treba
MODEL_PATH = ensure_model_downloaded()

# -----------------------
# Defaults & settings
# -----------------------
EMB_MODEL_NAME_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim


EMB_DIM_DEFAULT = 384
MAX_CHARS = 900             # umesto 1400
CHUNK_OVERLAP_CHARS = 120   # umesto 200
TOP_K = 8                   # umesto 12
MMR_K = 4                   # umesto 6
MMR_LAMBDA = 0.55           # relevantno nego raznoliko >> 0.55 >> 0.4


RANDOM_SEED = int(os.getenv("RAG_SEED", "42"))
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -----------------------
# Data classes & helpers
# -----------------------
@dataclass
class Chunk:
    doc_id: str
    page: int
    text: str
    source_path: str
    meta: Dict[str, Any]
    def as_dict(self): return asdict(self)

def normalize_ws(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"­", "", s)           # soft hyphen
    s = re.sub(r"-\n", "", s)         # hyphenated line breaks
    s = re.sub(r"\n{2,}", "\n", s)
    s = normalize_ws(s)
    return s

def sentence_split(s: str) -> List[str]:
    # lightweight sentence splitting for CJK/latin scripts with common abbreviations
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-ZĆČŠĐŽ])", s)
    merged, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(buf) + len(p) < 200:
            buf = (buf + " " + p).strip()
        else:
            if buf: merged.append(buf)
            buf = p
    if buf: merged.append(buf)
    return merged

def chunk_sentences(sentences: List[str], max_chars=MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[str]:
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf: chunks.append(buf)
            buf = (chunks[-1][-overlap:] + " " + s).strip() if (overlap > 0 and chunks) else s
    if buf: chunks.append(buf)
    return chunks

def load_pdf(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i+1, clean_text(txt)))
    return out

def build_chunks_from_pdfs(pdf_paths: List[Path]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for p in pdf_paths:
        logging.info(f"Ingesting {p} ...")
        for page_no, raw in load_pdf(p):
            if not raw.strip(): continue
            sents = sentence_split(raw)
            for idx, ch in enumerate(chunk_sentences(sents)):
                all_chunks.append(Chunk(
                    doc_id=p.stem, page=page_no, text=ch, source_path=str(p),
                    meta={"chunk_idx": idx, "filename": p.name}
                ))
    logging.info(f"Built {len(all_chunks)} chunks")
    return all_chunks

# -----------------------
# Embeddings
# -----------------------
class STEmbedder:
    def __init__(self, model_name: str):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed. Install it or use a GGUF embedding model.")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype("float32")

class GGUFEmbedder:
    """
    GGUF embedding model via llama.cpp (e.g., bge-small-en-v1.5-q4_0.gguf, dim=384).
    """
    def __init__(self, model_path: str, n_threads: int = None, n_gpu_layers: int = 0, dim_hint: int = EMB_DIM_DEFAULT):
        self.llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=1024,
            n_threads=n_threads or max(1, os.cpu_count()//2),
            n_gpu_layers=n_gpu_layers,
        )
        self.dim = dim_hint
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            e = self.llm.embed(t)
            v = np.array(e, dtype="float32")
            # Normalize for cosine
            v /= (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.vstack(vecs).astype("float32")

# -----------------------
# FAISS vector store
# -----------------------
class FaissStore:
    def __init__(self, dim: int, db_dir: Path):
        self.dim, self.db_dir = dim, db_dir
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
    def build(self, embeddings: np.ndarray, chunks: List[Chunk]):
        assert embeddings.shape[1] == self.dim, f"Emb dim {embeddings.shape[1]} != {self.dim}"
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.chunks = chunks
    def save(self):
        self.db_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.db_dir / "index.faiss"))
        with open(self.db_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks: f.write(json.dumps(c.as_dict(), ensure_ascii=False) + "\n")
    def load(self):
        self.index = faiss.read_index(str(self.db_dir / "index.faiss"))
        self.chunks = []
        with open(self.db_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f: self.chunks.append(Chunk(**json.loads(line)))
    def search(self, q_emb: np.ndarray, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(q_emb, top_k)
    def reconstruct_n(self):
        return self.index.reconstruct_n(0, self.index.ntotal)

def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, lambda_mult: float = MMR_LAMBDA, top_k: int = MMR_K) -> List[int]:
    selected = []
    sim_to_query = (doc_vecs @ query_vec.reshape(-1, 1)).flatten()
    while len(selected) < min(top_k, len(doc_vecs)):
        if not selected:
            selected.append(int(np.argmax(sim_to_query)))
            continue
        sel_mat = doc_vecs[selected]
        max_sim_selected = (doc_vecs @ sel_mat.T).max(axis=1)
        scores = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_selected
        scores[selected] = -1e9
        selected.append(int(np.argmax(scores)))
    return selected

# -----------------------
# LLM (llama.cpp)
# -----------------------
SYS_PROMPT = (
    "Ti si pažljiv asistent koji odgovara samo na osnovu datog konteksta. "
    "Ako nema dovoljno informacija u kontekstu, reci to jasno."
)

def build_prompt(question: str, contexts: List[Chunk]) -> str:
    header = (
        "Odgovori stručno i koncizno koristeći samo DONJI KONTEKST.\n"
        "Ako nešto nije u kontekstu, reci da nema dovoljno informacija.\n"
        "Na kraju navedi kratke izvore: [naziv.pdf, strana X].\n\n"
        f"PITANJE:\n{question}\n\nKONTEKST:\n"
    )
    ctx_txt = ""
    for i, ch in enumerate(contexts, 1):
        src = Path(ch.source_path).name
        ctx_txt += f"[{i}] ({src}, p.{ch.page})\n{ch.text}\n\n"
    return header + ctx_txt

def format_citations(contexts: List[Chunk]) -> str:
    seen = set()
    cites = []
    for ch in contexts:
        key = (Path(ch.source_path).name, ch.page)
        if key not in seen:
            seen.add(key)
            cites.append(f"[{key[0]}, strana {key[1]}]")
    return " Izvori: " + "; ".join(cites[:3]) if cites else ""


class LocalLLM:
    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_threads: int = None, ctx_size: int = 4096):
        from llama_cpp import Llama
        # Forsiramo "completion" API i gasimo chat-format da bismo izbegli template probleme
        self.llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or max(1, os.cpu_count()//2),
            chat_format=None,
            n_batch=256  # malo veći batch radi brzine na CPU
        )

    @retry(wait=wait_exponential_jitter(initial=1, max=8), stop=stop_after_attempt(3))
    def generate(self, question: str, contexts: List[Chunk], temperature: float = 0.15, max_tokens: int = 1024) -> str:
        # Instrukcije direktno u [INST]…[/INST]; forsiramo srpski (latinica)
        inst = (
            "Odgovaraj ISKLJUČIVO na srpskom jeziku (latinica). "
            "Koristi SAMO dati KONTEKST. Ako nema dovoljno informacija u kontekstu, reci to jasno. "
            "Odgovori sažeto i celovito (3–6 rečenica). "
            "Na kraju prikaži 2–3 izvora u formatu: [naziv.pdf, strana X]."
        )
        prompt = f"[INST] {inst}\n\n{build_prompt(question, contexts)} [/INST]"

        out = self.llm.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.05,
            max_tokens=max_tokens,
            stop=["</s>"]  # EOS; sprečava balavljenje, ali dopušta kompletan odgovor
        )
        return out["choices"][0]["text"].strip()

# -----------------------
# Orchestrator
# -----------------------
class RAGPipeline:
    def __init__(self, db_dir: Path, embedder, llm: LocalLLM):
        self.db_dir = db_dir
        self.embedder = embedder
        self.llm = llm
        self.store = FaissStore(getattr(embedder, "dim", EMB_DIM_DEFAULT), db_dir)

    def index_dir(self, data_dir: Path):
        pdfs = [p for p in Path(data_dir).glob("**/*") if p.suffix.lower() == ".pdf"]
        if not pdfs:
            print(f"[!] Nema PDF fajlova u: {data_dir}")
        chunks = build_chunks_from_pdfs(pdfs)
        texts = [c.text for c in chunks]
        print("[*] Kreiram ugrađivanja (embeddings)...")
        X = self.embedder.embed_texts(texts)
        self.store.build(X, chunks)
        self.store.save()
        print(f"[OK] Indeksirano {len(chunks)} delova teksta u {self.db_dir}")

    def ask(self, question: str, top_k: int = TOP_K, mmr_k: int = MMR_K) -> str:
        self.store.load()
        q = self.embedder.embed_texts([question])  # (1, d)
        D, I = self.store.search(q, top_k=top_k)
        docs = [self.store.chunks[i] for i in I[0]]
        all_vecs = self.store.reconstruct_n()
        retrieved_vecs = all_vecs[I[0]]
        sel = mmr(q[0], retrieved_vecs, lambda_mult=MMR_LAMBDA, top_k=mmr_k)
        contexts = [docs[i] for i in sel]
        answer = self.llm.generate(question, contexts)
        return answer + "\n\n" + format_citations(contexts)

# -----------------------
# Config IO
# -----------------------
def load_or_create_config(db_dir: Path) -> Dict[str, Any]:
    cfg_path = db_dir / "rag_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n=== Prvo podešavanje (sačuvaćemo u rag_config.json) ===")
    data_dir = input("Putanja do foldera sa PDF-ovima (npr. D:\\Dokumenti\\zakoni): ").strip('" ').strip()
    if not data_dir:
        print("[!] Morate navesti putanju do PDF-ova."); sys.exit(1)

    # Choose embedder
    use_st = False
    if _HAS_ST:
        ans = input(f"Koristi sentence-transformers za embeddings? [Y/n] ").strip().lower()
        use_st = (ans != "n")

    embedder_cfg = {}
    if use_st:
        embedder_cfg = {"type": "st", "model_name": EMB_MODEL_NAME_DEFAULT, "dim": EMB_DIM_DEFAULT}
    else:
        print("\nZa 100% lokalno, navedite GGUF embedding model (npr. bge-small-en-v1.5-q4_0.gguf).")
        embed_gguf = input("Putanja do GGUF embedding modela: ").strip('" ').strip()
        dim_text = input(f"Dimenzija embeddinga (podrazumevano {EMB_DIM_DEFAULT}): ").strip()
        dim = int(dim_text) if dim_text else EMB_DIM_DEFAULT
        embedder_cfg = {"type": "gguf", "model_path": embed_gguf, "dim": dim}

    print("\nOdaberite lokalni LLM (GGUF) za odgovaranje (npr. mistral-7b-instruct-v0.2.Q4_K_M.gguf).")
    llm_path = input("Putanja do LLM GGUF modela: ").strip('" ').strip()
    ctx_text = input("Context size (tokeni, podrazumevano 4096): ").strip()
    ctx_size = int(ctx_text) if ctx_text else 4096
    gpu_layers_text = input("n_gpu_layers (0 za CPU, npr. 40 za GPU): ").strip()
    n_gpu_layers = int(gpu_layers_text) if gpu_layers_text else 0

    cfg = {
        "data_dir": data_dir,
        "db_dir": str(db_dir),
        "embedder": embedder_cfg,
        "llm": {"model_path": llm_path, "ctx_size": ctx_size, "n_gpu_layers": n_gpu_layers}
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] Sačuvano: {cfg_path}")
    return cfg

def make_embedder(embed_cfg: Dict[str, Any]):
    if embed_cfg.get("type") == "st":
        model_name = embed_cfg.get("model_name", EMB_MODEL_NAME_DEFAULT)
        return STEmbedder(model_name)
    elif embed_cfg.get("type") == "gguf":
        model_path = embed_cfg.get("model_path", "")
        dim = int(embed_cfg.get("dim", EMB_DIM_DEFAULT))
        return GGUFEmbedder(model_path, dim_hint=dim)
    else:
        raise RuntimeError("Nepoznata vrsta embeddera.")

def make_llm(llm_cfg: Dict[str, Any]) -> LocalLLM:
    return LocalLLM(
        model_path=llm_cfg["model_path"],
        ctx_size=int(llm_cfg.get("ctx_size", 4096)),
        n_gpu_layers=int(llm_cfg.get("n_gpu_layers", 0))
    )

# -----------------------
# Interactive main
# -----------------------
def interactive_loop():
    print("\n=== RAG Chat (lokalni modeli) ===")
    # Choose / create DB dir
    default_db = Path("./rag_index")
    db_dir_text = input(f"Folder za indeks (podrazumevano {default_db}): ").strip()
    db_dir = Path(db_dir_text) if db_dir_text else default_db

    cfg = load_or_create_config(db_dir)
    embedder = make_embedder(cfg["embedder"])
    llm = make_llm(cfg["llm"])

    rag = RAGPipeline(db_dir=db_dir, embedder=embedder, llm=llm)

    # Build index if missing
    index_path = db_dir / "index.faiss"
    if not index_path.exists():
        print("\n[*] Indeks ne postoji – kreiram ga prvi put...")
        rag.index_dir(Path(cfg["data_dir"]))
    else:
        print(f"\n[OK] Indeks pronađen: {index_path}")

    print("\nUnesite pitanje (Ctrl+C za izlaz).")
    while True:
        try:
            q = input("\n?> ").strip()
            if not q:
                continue
            if q.lower() in {"/reindex", "/reload"}:
                print("[*] Reindeksiram ...")
                rag.index_dir(Path(cfg["data_dir"]))
                continue
            ans = rag.ask(q)
            print("\n" + "="*80)
            print(ans)
            print("="*80)
        except KeyboardInterrupt:
            print("\n[bye] Izlaz.")
            break
        except Exception as e:
            print(f"[ERR] {e}")

if __name__ == "__main__":
    # Optional: allow `rag_chat.py --noninteractive` or similar in the future
    try:
        interactive_loop()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


########################################################################################################################
########################################################################################################################
########################################################################################################################
### >> RUNNING THE SCRIPT >>
########################################################################################################################
########################################################################################################################
########################################################################################################################

# 1) index folder
# >> index

# 2) Path to the data >> Zakon o telekomunikacijama.pdf
# >> D:\Programi_PYTHON\proj_Yet\Virtual_Buddy\

# 3) Model folder - this mistral model was chosen because I already used it for different example
# >> D:\Programi_PYTHON\proj_Yet\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf

# 4) Max token limit
# >> 4096

# 5) Device type 0-CPU
# >> 0


# 6) Pitanja
# >> Ko izdaje dozvole?
# >> Sta je predmet Zakona o telekomunikacijama i koja pitanja on uredjuje?