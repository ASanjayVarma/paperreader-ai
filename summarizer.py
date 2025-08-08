# summarizer.py
import re
from typing import Dict
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import textwrap

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

# Helpers
SECTION_HEADINGS = [
    r"abstract",
    r"introduction",
    r"background",
    r"methods?",
    r"methodology",
    r"materials and methods",
    r"results?",
    r"findings",
    r"discussion",
    r"conclusion",
    r"concluding remarks",
]

SECTION_PATTERNS = r"(?im)^\s*(?P<header>" + r"|".join([
    r"Abstract", r"ABSTRACT",
    r"Introduction", r"INTRODUCTION",
    r"Methods", r"METHODS", r"Methodology", r"METHODODOLOGY",
    r"Materials and Methods", r"RESULTS", r"Results",
    r"Discussion", r"CONCLUSION", r"CONCLUSIONS", r"Conclusion"
]) + r")\b.*$"

def split_into_sections(text: str) -> Dict[str, str]:
    """
    Splits a paper's text into sections by heading lines.
    Returns dict with keys lowercased like 'abstract', 'methods', 'results', 'conclusion'.
    Heuristic-based; may not be perfect.
    """
    # Normalize newlines
    lines = text.splitlines()
    headings_idx = []
    for i, line in enumerate(lines):
        # A heading line is short and contains one of our keywords
        stripped = line.strip()
        if len(stripped) < 100:  # heuristic: headings tend to be short
            for h in ["abstract", "introduction", "methods", "methodology", "materials and methods", "results", "discussion", "conclusion"]:
                if re.search(rf"^\s*{h}\b", stripped, flags=re.I):
                    headings_idx.append((i, stripped.lower()))
                    break

    # Add artificial start and end
    headings_idx = sorted(headings_idx, key=lambda x: x[0])
    sections = {}
    if not headings_idx:
        # fallback: attempt to extract first 2000 chars as abstract-like
        sections["abstract"] = text[:2000]
        return sections

    for idx, (line_no, header) in enumerate(headings_idx):
        start = line_no + 1
        end = len(lines)
        if idx + 1 < len(headings_idx):
            end = headings_idx[idx + 1][0]
        body = "\n".join(lines[start:end]).strip()
        # map header to our keys
        if "abstract" in header:
            sections["abstract"] = body
        elif "method" in header or "materials and methods" in header:
            sections["methods"] = body
        elif "result" in header or "findings" in header:
            sections["results"] = body
        elif "discussion" in header or "conclusion" in header:
            # If conclusion exists, append or set
            prev = sections.get("conclusion", "")
            sections["conclusion"] = (prev + "\n\n" + body).strip() if prev else body
        else:
            # store generic if needed
            sections[header] = body

    # Attempt to ensure keys exist
    for k in ["abstract", "methods", "results", "conclusion"]:
        sections.setdefault(k, "")

    return sections

def chunk_text(text: str, max_chars: int = 1000):
    """
    Yield chunks of text <= max_chars, try to split on sentence boundaries.
    """
    text = text.strip()
    if len(text) <= max_chars:
        yield text
        return
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cur = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) + 1 <= max_chars:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            yield " ".join(cur)
            cur = [s]
            cur_len = len(s) + 1
    if cur:
        yield " ".join(cur)

class Summarizer:
    def __init__(self, mode="hf", hf_model="facebook/bart-large-cnn", openai_api_key=None):
        """
        mode: 'hf' or 'openai'
        hf_model: name of HF seq2seq model
        openai_api_key: if mode == 'openai'
        """
        self.mode = mode
        self.hf_model_name = hf_model
        self.hf_summarizer = None
        if mode == "hf":
            # Load tokenizer & model lazily when needed (expensive)
            self._load_hf_model()
        elif mode == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            if openai_api_key:
                openai.api_key = openai_api_key

    def _load_hf_model(self):
        if self.hf_summarizer is None:
            # Using transformers pipeline
            self.hf_summarizer = pipeline("summarization", model=self.hf_model_name, framework="pt", device=-1)

    def detect_sections(self, full_text: str):
        return split_into_sections(full_text)

    def summarize_section(self, section_text: str, max_length: int = 120) -> str:
        if not section_text or section_text.strip() == "":
            return "(No content found)"
        if self.mode == "hf":
            self._load_hf_model()
            # Chunk and summarize then combine
            chunks = list(chunk_text(section_text, max_chars=1000))
            summaries = []
            for chunk in chunks:
                # ensure max_length tokens (roughly characters -> tokens isn't exact)
                out = self.hf_summarizer(chunk, max_length=max_length, min_length=20, do_sample=False)[0]["summary_text"]
                summaries.append(out.strip())
            # Post-process: join and possibly compress
            joined = " ".join(summaries)
            # If joined is long, truncate elegantly
            if len(joined) > 1000:
                return textwrap.shorten(joined, width=1000, placeholder=" ...")
            return joined
        else:
            # Use OpenAI ChatCompletion (gpt-4o-mini-ish). We craft a prompt.
            prompt = (
                "You are a helpful assistant that rewrites academic sections into simple, plain-English summaries for a non-expert audience.\n\n"
                f"Section text:\n{section_text}\n\n"
                f"Produce a short, clear summary (max ~{max_length} tokens) in simple English. Use bullet points if suitable."
            )
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_length,
                temperature=0.2
            )
            summary = response["choices"][0]["message"]["content"].strip()
            return summary

    def summarize_tldr(self, full_text: str, max_length: int = 60) -> str:
        """
        Create a 2-3 sentence TL;DR summarizing the whole paper.
        """
        if self.mode == "hf":
            self._load_hf_model()
            # Chunk a leading part of the paper for speed
            snippet = full_text[:4000]
            out = self.hf_summarizer(snippet, max_length=max_length, min_length=20, do_sample=False)[0]["summary_text"]
            return out.strip()
        else:
            prompt = (
                "Read the following research paper text and produce a 2–3 sentence TL;DR in plain English for a non-expert:\n\n"
                f"{full_text[:4000]}"
            )
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_length,
                temperature=0.2
            )
            return response["choices"][0]["message"]["content"].strip()
