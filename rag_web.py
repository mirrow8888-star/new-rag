# ═══ 多策略 RAG 文件問答（網頁版）· Streamlit + Groq + ChromaDB + LINE / Telegram / Email 投放 ═══
# 執行：streamlit run rag_web.py
# 安裝：pip install -r requirements.txt
#
# 投放管道（側欄「📣 回答投放」分頁）：
#   • LINE     ：貼上 Channel Access Token → Broadcast 給所有好友（消耗每月訊息額度）
#   • Telegram ：貼上 Bot Token + Chat ID → sendMessage 給指定對話
#                Chat ID 可按「偵測」自動抓取（需先在 Telegram 對你的 Bot 說過一句話）
#                勾「超長回答自動分段多則」可突破單則上限，依段落／句尾切並標上（n/m）
#   • Email    ：貼上 Resend API Key + 收件信箱 → 寄出最新回答，並附上完整問答紀錄（Word / CSV）
#                預設寄件者 onboarding@resend.dev 僅能寄到你註冊 Resend 的信箱；驗證網域後可改自己的寄件者
#   • 模板     ：自訂訊息格式，{q} {ans} {doc} {strategy} {model} {k} {time} {date}（LINE / Telegram 共用）
from __future__ import annotations

import base64
import csv
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from html import escape as h_esc
from itertools import count
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
import requests
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
from linebot.v3.messaging.exceptions import ApiException
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")   # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000


LINE_LIMIT, LINE_HOST = 500, "https://api.line.me"   # LINE 訊息字數上限（可自行調整）
TG_LIMIT, TG_HOST = 3500, "https://api.telegram.org"  # Telegram 單則上限 4096，留餘裕
TG_PAUSE = 0.5                                        # 分段多則之間的間隔（秒），避免 429
RESEND_URL = "https://api.resend.com/emails"
MAIL_FROM = "Acme <onboarding@resend.dev>"            # Resend 測試寄件者；驗證網域後可改成自己的信箱
MAIL_ATTACH = {"📄 Word + 📊 CSV": ("docx", "csv"), "📄 只附 Word": ("docx",),
               "📊 只附 CSV": ("csv",), "🚫 不附檔案": ()}   # 下拉選單 → 要附上的報告格式

# ── 投放訊息模板 ──
TPL_DEFAULT = "❓ {q}\n\n{ans}"
TPL_HINT = ("可用欄位：{q} 問題 · {ans} 回答 · {doc} 文件名稱 · {strategy} 策略 · "
            "{model} 模型 · {k} 片段數 · {time} 時間 · {date} 日期")
TPL_SAMPLE = {"q": "這份文件的重點是什麼？", "ans": "重點一：適用範圍為全體同仁。\n重點二：申請需於七日前提出。",
              "doc": "員工手冊.pdf", "strategy": "⚡ 混合搜尋", "model": "openai/gpt-oss-20b",
              "k": 4, "time": "14:30", "date": "2026-01-01"}


STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


class RAG:
    """嵌入模型 + 向量庫 + 目前載入的單一文件。"""

    def __init__(self):
        self._embedder, self._lock = None, threading.Lock()
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()  # 記憶體模式，關閉程式即清空
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        self.doc: Doc | None = None

    @property
    def ready(self):
        return self._embedder is not None

    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                    m.max_seq_length = min(512, m.max_seq_length or 512)
                    self._embedder = m
        return self._embedder

    # ── 文件載入 ──
    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"):
            raise ValueError("僅支援 PDF / DOCX")
        segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, path.name, [p[0] for p in pairs], [p[1] for p in pairs])
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.drop(doc)
            raise
        self.drop(self.doc)  # 新文件成功後才釋放舊文件
        self.doc = doc
        return f"✓ {doc.name}\n{len(doc.chunks)} 片段 · {time.time() - t0:.1f}s"

    def drop(self, doc):
        if doc is None:
            return
        for c in (self.col, self.ccol):
            try:
                c.delete(where={"sid": doc.sid})
            except Exception:
                pass

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        res = (self.ccol if child else self.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 耗時說明)。"""
        doc = self.doc
        if doc is None:
            return "請先載入 PDF 或 DOCX 文件。", "", ""
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", src, ""
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（檢索結果請看「檢索片段」分頁）", src, ""
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        return ans or "（模型未回傳內容）", src, f"{llm.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s"


# ── 匯出：Word / CSV ──
def export_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟中文不亂碼
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def export_docx(records, path, with_src=True):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Microsoft JhengHei", Pt(11)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    doc.save(path)


def plain(md: str) -> str:
    """Tk 文字框不渲染 Markdown：移除粗體標記，標題轉成一般行。"""
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


# ═══════════════ 投放：訊息組裝 ═══════════════
def clean_answer(ans):
    """回答轉純文字：移除 Markdown 標記與 [片段n] 標註、壓掉多餘空行。"""
    body = re.sub(r"[ \t]*\[片段[^\]]*\]", "", plain(ans))
    body = re.sub(r"[ \t]+(?=\n|$)", "", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def render_tpl(tpl, fields):
    """套用模板：{欄位} 以 fields 取代；未知欄位原樣保留，不會拋 KeyError。"""
    out = re.sub(r"\{(\w+)\}", lambda m: str(fields.get(m.group(1), m.group(0))), tpl.strip() or TPL_DEFAULT)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def fit(text, limit):
    """單則模式：超過上限就截斷，盡量切在句尾。"""
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    end = max(cut.rfind(c) for c in "\n。！？；")
    return (cut[:end + 1] if end > limit * 0.6 else cut).rstrip() + "…"


def split_msgs(text, limit):
    """分段模式：切成多則，優先在段落邊界、其次句尾、最後硬切；多於一則時標上（n/m）。"""
    room = max(60, limit - 12)   # 預留 (n/m) 標記空間
    units = []                   # [(片段文字, 與前一段的連接符)]
    for para in re.split(r"\n{2,}", text.strip()):
        para = para.strip()
        if not para:
            continue
        if len(para) <= room:
            units.append((para, "\n\n"))
            continue
        first = True             # 段落被迫拆開：第一塊沿用段落邊界，其餘直接接續
        for sent in (s.strip() for s in re.split(r"(?<=[。！？!?；;])|\n", para)):
            if not sent:
                continue
            for i in range(0, len(sent), room):
                units.append((sent[i:i + room], "\n\n" if first else ""))
                first = False
    parts, buf = [], ""
    for seg, sep in units:
        if buf and len(buf) + len(sep) + len(seg) > room:
            parts.append(buf)
            buf = seg
        else:
            buf = f"{buf}{sep}{seg}" if buf else seg
    if buf:
        parts.append(buf)
    parts = parts or [text[:room]]
    return parts if len(parts) == 1 else [f"{p}\n\n（{i}/{len(parts)}）" for i, p in enumerate(parts, 1)]


# ── LINE Bot ──
def line_send(token, text, host=LINE_HOST):
    """Broadcast 給所有加入此官方帳號的好友（會消耗每月訊息額度）。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        MessagingApi(c).broadcast(BroadcastRequest(messages=[TextMessage(text=text)]))


def line_check(token, host=LINE_HOST):
    """驗證 Token，回傳官方帳號名稱。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        return MessagingApi(c).get_bot_info().display_name


def line_err(e):
    if isinstance(e, ApiException):
        hint = {400: "請求格式錯誤", 401: "Token 無效或已過期", 403: "權限不足", 429: "超過訊息額度或請求過於頻繁"}.get(e.status, "")
        return f"HTTP {e.status} {hint}".strip()
    return f"{type(e).__name__}: {e}"


# ── Telegram Bot ──
class TgError(RuntimeError):
    """Telegram API 回傳 ok=false，或網路層失敗。"""


def _tg(token, method, data=None, timeout=20, host=TG_HOST):
    token = (token or "").strip()
    if not token:
        raise TgError("未設定 Bot Token")
    url = f"{host}/bot{token}/{method}"
    try:
        r = requests.post(url, data=data, timeout=timeout) if data else requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise TgError(f"網路錯誤（{type(e).__name__}）") from e
    try:
        j = r.json()
    except ValueError:
        raise TgError(f"HTTP {r.status_code}：非 JSON 回應")
    if not j.get("ok"):
        hint = {400: "請求錯誤（Chat ID 可能不正確）", 401: "Bot Token 無效", 403: "機器人被封鎖或尚未與該對話互動",
                404: "Token 或 API 路徑錯誤", 429: "請求過於頻繁"}.get(r.status_code, "")
        desc = j.get("description") or f"HTTP {r.status_code}"
        raise TgError(f"{desc}{' · ' + hint if hint else ''}")
    return j.get("result")


def tg_check(token):
    """驗證 Bot Token，回傳 @username。"""
    me = _tg(token, "getMe") or {}
    name = me.get("username")
    return f"@{name}" if name else (me.get("first_name") or "Bot")


def tg_chats(token):
    """從 getUpdates 取出曾與機器人互動的對話：{chat_id: 顯示名稱}。
    需先在 Telegram 對 Bot 發過訊息；若已設定 webhook，Telegram 會拒絕 getUpdates。"""
    out = {}
    for u in _tg(token, "getUpdates") or []:
        for key in ("message", "edited_message", "channel_post", "edited_channel_post", "my_chat_member"):
            m = u.get(key) or {}
            c = m.get("chat") or {}
            cid = c.get("id")
            if cid is None:
                continue
            name = (c.get("title") or " ".join(x for x in (c.get("first_name"), c.get("last_name")) if x)
                    or c.get("username") or "未命名")
            out[str(cid)] = f"{name}·{c.get('type', '')}"
    return out


def tg_send(token, chat_id, text):
    """傳送純文字訊息到指定 Chat（Telegram 單則上限 4096 字）。"""
    chat_id = (chat_id or "").strip()
    if not chat_id:
        raise TgError("未設定 Chat ID")
    _tg(token, "sendMessage", {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"})


def tg_send_many(token, chat_id, texts, pause=TG_PAUSE):
    """依序傳送多則；中途失敗會回報已成功的則數。"""
    for i, t in enumerate(texts):
        if i:
            time.sleep(pause)
        try:
            tg_send(token, chat_id, t)
        except TgError as e:
            raise TgError(f"第 {i + 1}/{len(texts)} 則失敗：{e}") from e


def tg_err(e):
    return str(e) if isinstance(e, TgError) else f"{type(e).__name__}: {e}"


# ── Email（Resend）──
class MailError(RuntimeError):
    """Resend API 回傳錯誤，或網路層失敗。"""


_EMAIL_RE = re.compile(r"[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+")


def mail_recipients(raw):
    """解析收件人：可用逗號、分號或空白分隔多個信箱；格式有誤時拋出 ValueError。"""
    items = [x for x in re.split(r"[,;\s]+", raw.strip()) if x]
    if not items:
        raise ValueError("請填入收件信箱")
    bad = [x for x in items if not _EMAIL_RE.fullmatch(x)]
    if bad:
        raise ValueError(f"信箱格式不正確：{bad[0]}")
    return items


def mail_key_ok(key):
    return bool(re.fullmatch(r"re_[A-Za-z0-9_\-]{10,}", key))


def build_reports(records, kinds, with_src):
    """把問答紀錄輸出成 Word / CSV，回傳 [(檔名, bytes)]。檔名用 ASCII，避免部分信箱處理中文附件名出錯。"""
    stamp, out = f"{datetime.now():%Y%m%d_%H%M%S}", []
    with tempfile.TemporaryDirectory() as d:
        if "docx" in kinds:
            path = Path(d) / f"RAG_QA_Report_{stamp}.docx"
            export_docx(records, str(path), with_src)
            out.append((path.name, path.read_bytes()))
        if "csv" in kinds:
            path = Path(d) / f"RAG_QA_Report_{stamp}.csv"
            export_csv(records, str(path))
            out.append((path.name, path.read_bytes()))
    return out


def mail_content(f, n_records, names):
    """依最新回答組出 (主旨, HTML 內文, 純文字內文)。f 為模板欄位 dict（q / ans / doc / strategy / model / date / time）。"""
    meta = f"📄 {f['doc']}　🧭 {f['strategy']}　🤖 {f['model']}　🕒 {f['date']} {f['time']}"
    attach = f"📎 附件：{'、'.join(names)}（完整問答紀錄，共 {n_records} 題）" if names else f"本次工作階段共 {n_records} 題問答"
    html = (
        "<div style=\"font-family:-apple-system,'Segoe UI','Microsoft JhengHei',sans-serif;max-width:640px;margin:auto;color:#1f2937\">"
        "<h2 style=\"color:#059669;margin-bottom:4px\">📚 RAG 文件問答結果</h2>"
        f"<p style=\"color:#6b7280;font-size:13px;margin-top:0\">{h_esc(meta)}</p>"
        f"<h3 style=\"margin-bottom:4px\">❓ 問題</h3><p style=\"margin-top:0\">{h_esc(f['q'])}</p>"
        "<h3 style=\"margin-bottom:4px\">💡 回答</h3>"
        f"<p style=\"margin-top:0;line-height:1.7\">{h_esc(f['ans']).replace(chr(10), '<br>')}</p>"
        "<hr style=\"border:none;border-top:1px solid #e5e7eb\">"
        f"<p style=\"font-size:13px;color:#6b7280\">{h_esc(attach)}</p></div>")
    text = f"📚 RAG 文件問答結果\n{meta}\n\n❓ 問題\n{f['q']}\n\n💡 回答\n{f['ans']}\n\n{attach}"
    return f"📚 RAG 問答結果｜{fit(f['q'], 40)}", html, text


def mail_send(api_key, sender, to, subject, html, text, attachments):
    """透過 Resend REST API 寄信（等同 resend.Emails.send），回傳郵件 id。"""
    payload = {"from": sender, "to": to, "subject": subject, "html": html, "text": text}
    if attachments:
        payload["attachments"] = [{"filename": n, "content": base64.b64encode(b).decode("ascii")} for n, b in attachments]
    try:
        r = requests.post(RESEND_URL, json=payload, timeout=60,
                          headers={"Authorization": f"Bearer {api_key}", "User-Agent": "rag-desktop/1.0"})
    except requests.RequestException as e:
        raise MailError(f"網路錯誤（{type(e).__name__}）") from e
    try:
        j = r.json()
    except ValueError:
        j = {}
    if r.status_code >= 300:
        hint = {401: "API Key 無效", 403: "寄件網域未驗證；使用 onboarding@resend.dev 時只能寄到你註冊 Resend 的信箱",
                422: "欄位格式錯誤", 429: "請求過於頻繁或超過額度"}.get(r.status_code, "")
        msg = j.get("message") if isinstance(j, dict) else None
        raise MailError(f"{msg or 'HTTP ' + str(r.status_code)}{' · ' + hint if hint else ''}")
    return j.get("id", "") if isinstance(j, dict) else ""


def mail_err(e):
    return str(e) if isinstance(e, (MailError, ValueError)) else f"{type(e).__name__}: {e}"


# ═══════════════ 網頁介面 ═══════════════
st.set_page_config(page_title="RAG 文件問答", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="⏳ 載入嵌入模型（首次需下載，約 470 MB）…")
def get_embedder():
    """嵌入模型全站共用一份，避免每位使用者各載入一次。"""
    return RAG().embedder()


@st.cache_data(ttl=600, show_spinner=False)
def cached_models(key):
    return list_models(key)


def init_state():
    S = st.session_state
    if "rag" not in S:
        rag = RAG()
        rag._embedder = get_embedder()
        S.rag, S.records, S.last, S.doc_sig, S.last_src = rag, [], None, None, ""
        S.msgs = {k: ("", "") for k in ("line", "tg", "mail")}   # 各管道最近一次狀態 (ok/err, 文字)
    return S


def tmp_bytes(writer, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = f.name
    try:
        writer(path)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def set_msg(ch, kind, text):
    st.session_state.msgs[ch] = (kind, text)


def show_msg(ch, hint=""):
    kind, text = st.session_state.msgs[ch]
    if text:
        (st.success if kind == "ok" else st.error)(text)
    elif hint:
        st.caption(hint)


def compose(ch, fields, tpl, tg_split):
    """套模板 → 依管道上限截斷或分段，回傳要送出的訊息清單。"""
    text = render_tpl(tpl, fields)
    if ch == "tg" and tg_split:
        return split_msgs(text, TG_LIMIT)
    return [fit(text, LINE_LIMIT if ch == "line" else TG_LIMIT)]


def push(ch, label, fn, errfmt, msgs):
    """送出訊息並把結果寫入狀態（成功／失敗皆不拋出）。"""
    n = sum(len(m) for m in msgs)
    try:
        with st.spinner(f"投放到 {label}…"):
            fn()
        txt = f"{len(msgs)} 則 · {n} 字" if len(msgs) > 1 else f"{n} 字"
        set_msg(ch, "ok", f"✓ 已投放到 {label}（{txt}） · {datetime.now():%H:%M:%S}")
        st.toast(f"已投放到 {label}", icon="✅")
    except Exception as e:  # noqa: BLE001
        set_msg(ch, "err", f"✗ {label} 投放失敗：{errfmt(e)}")


def mail_fields(S):
    """最新一則回答的欄位；若尚無有效回答，退而使用最後一筆問答紀錄。"""
    if S.last:
        return S.last
    r, now = S.records[-1], datetime.now()
    return {"q": r["問題"], "ans": clean_answer(r["回答"]), "doc": r["文件"], "strategy": r["策略"],
            "model": r["模型"], "time": f"{now:%H:%M}", "date": f"{now:%Y-%m-%d}"}


def send_mail(S, key, to_raw, sender, attach):
    kinds = MAIL_ATTACH[attach]
    if not S.records:
        return set_msg("mail", "err", "⚠ 尚無問答，請先提問並取得回答")
    if not key:
        return set_msg("mail", "err", "⚠ 請先貼上 Resend API Key")
    if not mail_key_ok(key):
        return set_msg("mail", "err", "✗ API Key 格式不正確（應以 re_ 開頭）")
    try:
        to = mail_recipients(to_raw)
        with st.spinner("產生報告並寄送中…"):
            files = build_reports(list(S.records), kinds, S.get("with_src", True))
            subject, html, text = mail_content(mail_fields(S), len(S.records), [n for n, _ in files])
            mail_send(key, sender or MAIL_FROM, to, subject, html, text, files)
        txt = f"已寄到 {'、'.join(to)}" + (f"（附件 {len(files)} 個）" if files else "")
        set_msg("mail", "ok", f"✓ {txt} · {datetime.now():%H:%M:%S}")
        st.toast(txt, icon="📧")
    except Exception as e:  # noqa: BLE001
        set_msg("mail", "err", f"✗ Email 寄送失敗：{mail_err(e)}")


# ── callback（在頁面重新繪製前執行，可安全修改 widget 的 session_state）──
def cb_detect_tg():
    S = st.session_state
    tok = S.get("tg_tok", "").strip()
    if not tok:
        return set_msg("tg", "err", "⚠ 請先填入 Bot Token")
    try:
        chats = tg_chats(tok)
    except Exception as e:  # noqa: BLE001
        return set_msg("tg", "err", f"✗ 偵測失敗：{tg_err(e)}")
    if not chats:
        return set_msg("tg", "err", "未找到對話：請先在 Telegram 對 Bot 發一句話，再按偵測")
    ids = list(chats)
    if S.get("tg_chat", "").strip() not in ids:
        S.tg_chat = ids[0]
    set_msg("tg", "ok", f"✓ 找到 {len(ids)} 個對話：" + "；".join(f"{i}（{n}）" for i, n in chats.items()))


def cb_reset_tpl():
    st.session_state.tpl = TPL_DEFAULT


def check(ch, label, fn, errfmt):
    try:
        set_msg(ch, "ok", f"✓ {label} 有效 · {fn() or label}")
    except Exception as e:  # noqa: BLE001
        set_msg(ch, "err", f"✗ 驗證失敗：{errfmt(e)}")


def main():
    S = init_state()

    # ── 側欄 ──
    with st.sidebar:
        st.title("📚 RAG 文件問答")
        st.caption("Groq × ChromaDB × LINE / Telegram / Email")

        with st.container(border=True):
            st.markdown("**🔑 Groq 模型**")
            key = st.text_input("Groq API Key", type="password", placeholder="gsk_…", label_visibility="collapsed")
            models = cached_models(key.strip()) if key.strip() else []
            if key.strip():
                st.caption(f"✓ Key 有效 · {len(models)} 個模型" if models else "✗ Key 無效或無法取得模型")
            model = st.selectbox("模型", models or MODELS, label_visibility="collapsed")

        with st.container(border=True):
            st.markdown("**📄 文件**")
            up = st.file_uploader("選擇 PDF / DOCX", type=["pdf", "docx"], label_visibility="collapsed")
            if up is not None and S.doc_sig != (up.name, up.size):
                with st.spinner("⏳ 解析並建立索引中…"):
                    p = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{Path(up.name).name}"
                    p.write_bytes(up.getvalue())
                    try:
                        S.rag.load(p)
                        S.doc_sig = (up.name, up.size)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"✗ {e}" if isinstance(e, ValueError) else f"✗ 載入失敗：{type(e).__name__}: {e}")
                    finally:
                        p.unlink(missing_ok=True)
            if S.rag.doc:
                st.success(f"✓ {S.rag.doc.name}\n\n{len(S.rag.doc.chunks)} 片段")
            else:
                st.caption("尚未載入文件")

        with st.container(border=True):
            st.markdown("**🧭 檢索設定**")
            strategy = st.selectbox("策略", list(STRATEGIES), format_func=lambda k: STRATEGIES[k][0], label_visibility="collapsed")
            st.caption(STRATEGIES[strategy][1])
            topk = st.slider("片段數量 Top-K", 1, 10, 4)

        with st.container(border=True):
            st.markdown("**📣 回答投放**")
            t_line, t_tg, t_mail, t_tpl = st.tabs(["💬 LINE", "✈️ TG", "📧 Mail", "⚙️ 模板"])

            with t_line:
                ltok = st.text_input("Channel Access Token", type="password", key="line_tok").strip()
                lauto = st.checkbox(f"回答後自動群發（≤{LINE_LIMIT} 字）", key="line_auto")
                if lauto:
                    st.warning("每次回答都會群發給所有好友，並消耗每月訊息額度。")
                c1, c2 = st.columns(2)
                if c1.button("✔ 驗證", key="line_check", disabled=not ltok):
                    check("line", "Token", lambda: line_check(ltok), line_err)
                if c2.button("📤 投放", key="line_send", disabled=not (ltok and S.last)):
                    push("line", "LINE", lambda: [line_send(ltok, m) for m in compose("line", S.last, S.get("tpl", TPL_DEFAULT), False)],
                         line_err, compose("line", S.last, S.get("tpl", TPL_DEFAULT), False))
                show_msg("line", "⚠ 尚未設定 Token" if not ltok else "Broadcast：群發給所有加入官方帳號的好友")

            with t_tg:
                ttok = st.text_input("Bot Token", type="password", key="tg_tok", placeholder="123456:ABC…").strip()
                tchat = st.text_input("Chat ID", key="tg_chat").strip()
                st.button("🔍 偵測 Chat ID", on_click=cb_detect_tg)
                tauto = st.checkbox(f"回答後自動傳送（≤{TG_LIMIT} 字）", key="tg_auto")
                tsplit = st.checkbox("超長回答自動分段多則", value=True, key="tg_split")
                c1, c2 = st.columns(2)
                if c1.button("✔ 驗證", key="tg_check", disabled=not ttok):
                    check("tg", "Token", lambda: tg_check(ttok), tg_err)
                if c2.button("📤 投放", key="tg_send", disabled=not (ttok and tchat and S.last)):
                    msgs = compose("tg", S.last, S.get("tpl", TPL_DEFAULT), tsplit)
                    push("tg", "Telegram", lambda: tg_send_many(ttok, tchat, msgs), tg_err, msgs)
                show_msg("tg", "偵測前請先在 Telegram 對你的 Bot 說一句話")

            with t_mail:
                mkey = st.text_input("Resend API Key", type="password", key="mail_key", placeholder="re_…").strip()
                mto = st.text_input("收件信箱（Your Gmail）", key="mail_to", placeholder="you@gmail.com（多個以逗號分隔）").strip()
                msender = st.text_input("寄件者", value=MAIL_FROM, key="mail_from").strip()
                mattach = st.selectbox("附件報告（完整問答紀錄）", list(MAIL_ATTACH), key="mail_attach")
                c1, c2 = st.columns(2)
                if c1.button("✔ 檢查", key="mail_check", disabled=not mkey):
                    try:
                        n = len(mail_recipients(mto))
                        if not mail_key_ok(mkey):
                            set_msg("mail", "err", "✗ API Key 格式不正確（應以 re_ 開頭）")
                        else:
                            set_msg("mail", "ok", f"✓ 格式正確 · {n} 位收件人（寄出時才會實際驗證 Key）")
                    except ValueError as e:
                        set_msg("mail", "err", f"✗ {e}")
                if c2.button("📧 寄送", key="mail_send", disabled=not (mkey and mto and S.records)):
                    send_mail(S, mkey, mto, msender, mattach)
                show_msg("mail", "信件內文為最新一則問答；附件為目前所有問答紀錄。預設寄件者 onboarding@resend.dev 只能寄到你註冊 Resend 的信箱。")

            with t_tpl:
                st.text_area("訊息模板（LINE / Telegram 共用）", value=TPL_DEFAULT, key="tpl", height=110)
                st.caption(TPL_HINT)
                c1, c2 = st.columns(2)
                prev = c1.toggle("👁 預覽", key="tpl_preview")
                c2.button("↺ 預設", on_click=cb_reset_tpl)
                if prev:
                    fields = S.last or TPL_SAMPLE
                    st.caption("（目前為範例資料）" if not S.last else "（使用最新回答）")
                    for ch, label in (("line", "💬 LINE"), ("tg", "✈️ Telegram")):
                        ms = compose(ch, fields, S.get("tpl", TPL_DEFAULT), tsplit)
                        st.markdown(f"**{label}**：{len(ms)} 則 · {sum(len(m) for m in ms)} 字")
                        st.text("\n\n＋＋＋ 下一則 ＋＋＋\n\n".join(ms))

        with st.container(border=True):
            st.markdown("**💾 匯出問答紀錄**")
            st.checkbox("Word 包含參考片段（匯出與 Email 附件共用）", value=True, key="with_src")
            if S.records:
                c1, c2 = st.columns(2)
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                c1.download_button("📄 Word", tmp_bytes(lambda p: export_docx(S.records, p, S.with_src), ".docx"),
                                   f"RAG問答紀錄_{stamp}.docx")
                c2.download_button("📊 CSV", tmp_bytes(lambda p: export_csv(S.records, p), ".csv"),
                                   f"RAG問答紀錄_{stamp}.csv")
            else:
                st.caption("目前沒有問答紀錄可匯出")

        if st.button("🗑 清除對話"):
            S.records, S.last, S.last_src = [], None, ""
            S.msgs = {k: ("", "") for k in S.msgs}
            st.rerun()

    # ── 主區 ──
    tab_chat, tab_src = st.tabs(["💬 對話", "🔎 檢索片段"])
    with tab_chat:
        if not S.records:
            st.info("請先在左側載入 PDF / DOCX，再於下方輸入問題。回答後可從側欄「📣 回答投放」送到 LINE、Telegram 或 Email（含 Word / CSV 報告）。")
        for r in S.records:
            with st.chat_message("user"):
                st.write(r["問題"])
            with st.chat_message("assistant"):
                st.markdown(r["回答"])
                st.caption(f"{r['策略']} · {r['模型']} · {r['時間']}")
    with tab_src:
        st.text(S.last_src or "尚無檢索結果")

    q = st.chat_input("輸入問題，Enter 送出")
    if q:
        q = q.strip()
        with st.spinner("思考中…"):
            try:
                ans, src, timing = S.rag.answer(q, strategy, topk, model, key)
            except Exception as e:  # noqa: BLE001
                ans, src, timing = f"✗ {type(e).__name__}: {e}", "", ""
        S.last_src = src
        S.records.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "文件": S.rag.doc.name if S.rag.doc else "", "問題": q, "回答": ans,
                          "策略": STRATEGIES[strategy][0], "模型": model or MODELS[0],
                          "參考片段": src.split("─" * 40, 1)[-1].strip()})
        if timing:  # timing 非空 = 模型確實產生了回答（排除未載入文件／無 Key／無結果等提示）
            now = datetime.now()
            S.last = {"q": " ".join(q.split()), "ans": clean_answer(ans), "doc": S.rag.doc.name,
                      "strategy": STRATEGIES[strategy][0], "model": model or MODELS[0], "k": topk,
                      "time": f"{now:%H:%M}", "date": f"{now:%Y-%m-%d}"}
            tpl = S.get("tpl", TPL_DEFAULT)
            if lauto and ltok:
                msgs = compose("line", S.last, tpl, False)
                push("line", "LINE", lambda: [line_send(ltok, m) for m in msgs], line_err, msgs)
            if tauto and ttok and tchat:
                msgs = compose("tg", S.last, tpl, tsplit)
                push("tg", "Telegram", lambda: tg_send_many(ttok, tchat, msgs), tg_err, msgs)
        st.rerun()


main()
