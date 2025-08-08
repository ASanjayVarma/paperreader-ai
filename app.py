# app.py
import streamlit as st
from pdf_utils import extract_text_from_pdf
from summarizer import Summarizer
from diagram import methods_to_flowchart_dot
from utils import make_txt_bytes, make_pdf_bytes
import time

st.set_page_config(page_title="PaperReader AI", layout="wide")

st.title("📄 PaperReader — Read · Summarize · Visualize")
st.markdown(
    "Upload a research paper PDF and get plain-English summaries of Abstract, Methods, Results, Conclusion plus a TL;DR. Optional flowchart from Methods."
)

# Sidebar: model options
st.sidebar.header("Settings")
model_mode = st.sidebar.selectbox("Summarization mode", ["offline (Hugging Face)", "openai (API)"])
if model_mode == "offline (Hugging Face)":
    hf_model_choice = st.sidebar.selectbox("HF model", ["facebook/bart-large-cnn", "t5-base"])
else:
    openai_key = st.sidebar.text_input("OpenAI API Key (sk-...)", type="password")

max_summary_len = st.sidebar.slider("Summary max tokens (per section)", 40, 300, 120)
tldr_len = st.sidebar.slider("TL;DR max tokens", 20, 80, 60)
generate_flowchart = st.sidebar.checkbox("Generate flowchart from Methods", value=True)
show_raw_extraction = st.sidebar.checkbox("Show raw extracted text", value=False)

uploader = st.file_uploader("Upload paper (PDF)", type=["pdf"])
process_btn = st.button("Process Paper")

if uploader and process_btn:
    start_ts = time.time()
    with st.spinner("Reading PDF..."):
        raw_text = extract_text_from_pdf(uploader)
    if show_raw_extraction:
        st.subheader("Raw extracted text (first 2000 chars)")
        st.text_area("raw", raw_text[:2000], height=200)
    # Init summarizer
    with st.spinner("Loading summarizer..."):
        summarizer = Summarizer(
            mode="openai" if model_mode.startswith("openai") else "hf",
            hf_model=hf_model_choice if model_mode.startswith("offline") else None,
            openai_api_key=(openai_key if model_mode.startswith("openai") else None)
        )
    st.success("Models ready")

    with st.spinner("Detecting sections..."):
        sections = summarizer.detect_sections(raw_text)

    # Summarize each section
    st.markdown("## Results")
    tldr_text = summarizer.summarize_tldr(raw_text, max_length=tldr_len)
    st.markdown("### TL;DR")
    st.info(tldr_text)

    cols = st.columns(2)
    left, right = cols

    def display_section(name, content):
        if not content or content.strip() == "":
            return f"(No {name} found)"
        summary = summarizer.summarize_section(content, max_length=max_summary_len)
        return summary, content

    # Abstract
    with left:
        st.subheader("Abstract")
        abstract = sections.get("abstract", "")
        summary, _ = display_section("Abstract", abstract)
        st.write(summary)

        st.subheader("Methods")
        methods = sections.get("methods", "")
        ms_summary, ms_raw = display_section("Methods", methods)
        st.write(ms_summary)

    with right:
        st.subheader("Results")
        results = sections.get("results", "")
        rs_summary, _ = display_section("Results", results)
        st.write(rs_summary)

        st.subheader("Conclusion / Discussion")
        conclusion = sections.get("conclusion", "")
        cs_summary, _ = display_section("Conclusion", conclusion)
        st.write(cs_summary)

    # Optional Flowchart
    if generate_flowchart and methods and methods.strip():
        with st.spinner("Generating flowchart..."):
            dot = methods_to_flowchart_dot(methods)
        st.subheader("Methods Flowchart")
        st.graphviz_chart(dot)

    # Download buttons: compile summaries
    full_summary = []
    full_summary.append("TL;DR:\n" + tldr_text + "\n\n")
    full_summary.append("Abstract Summary:\n" + (ms := (display_section("Abstract", abstract)[0] if abstract else "(none)")) + "\n\n")
    full_summary.append("Methods Summary:\n" + (display_section("Methods", methods)[0] if methods else "(none)") + "\n\n")
    full_summary.append("Results Summary:\n" + (display_section("Results", results)[0] if results else "(none)") + "\n\n")
    full_summary.append("Conclusion Summary:\n" + (display_section("Conclusion", conclusion)[0] if conclusion else "(none)") + "\n\n")

    text_blob = "\n".join(full_summary)
    st.download_button("Download summaries (.txt)", data=make_txt_bytes(text_blob), file_name="paper_summaries.txt", mime="text/plain")

    pdf_bytes = make_pdf_bytes(text_blob, title="Paper Summaries")
    st.download_button("Download summaries (.pdf)", data=pdf_bytes, file_name="paper_summaries.pdf", mime="application/pdf")

    end_ts = time.time()
    st.sidebar.success(f"Processed in {end_ts - start_ts:.1f}s")
else:
    st.info("Upload a PDF and press **Process Paper**")
