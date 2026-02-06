# ===============================
# INSTALLS
# ===============================
# !pip install gradio transformers sentence-transformers tavily-python wikipedia diskcache torch --quiet

# ===============================
# IMPORTS
# ===============================
import os, re
from urllib.parse import urlparse
from collections import Counter

import gradio as gr
import torch
import wikipedia

from tavily import TavilyClient
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diskcache import Cache

# ===============================
# API KEY
# ===============================
os.environ["TAVILY_API_KEY"] = " "

# ===============================
# INITIALIZATION
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

tavily = TavilyClient()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

cache = Cache("./tavily_cache")

# ===============================
# SEARCH FUNCTIONS
# ===============================
def safe_tavily_search(query, max_results=5):
    try:
        if query in cache:
            return cache[query]
        res = tavily.search(query=query, max_results=max_results)
        results = [{
            "url": r["url"],
            "title": r["title"],
            "content": r.get("content", "")
        } for r in res.get("results", [])]
        cache[query] = results
        return results
    except:
        return []

def wiki_search(query):
    out = []
    try:
        for title in wikipedia.search(query, results=2):
            try:
                page = wikipedia.page(title, auto_suggest=False)
                out.append({
                    "url": page.url,
                    "title": page.title,
                    "content": page.summary[:1200]
                })
            except:
                pass
    except:
        pass
    return out

# ===============================
# EVIDENCE RETRIEVAL
# ===============================
def retrieve_evidence(claim):
    sources, seen = [], set()
    queries = [
        claim,
        f"fact check {claim}",
        f"{claim} explained",
        f"{claim} controversy"
    ]

    for q in queries:
        for src in safe_tavily_search(q) + wiki_search(q):
            if src["url"] not in seen:
                seen.add(src["url"])
                sources.append(src)

    if not sources:
        return []

    scores = reranker.predict([[claim, s["content"][:400]] for s in sources])
    ranked = sorted(
        [{**s, "score": float(sc)} for s, sc in zip(sources, scores)],
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:4]

# ===============================
# RULE-BASED FACT VERIFICATION
# ===============================
def rule_based_verdict(claim, evidence):
    claim_lower = claim.lower()

    for e in evidence:
        text = e["content"].lower()

        claim_pres = re.search(r"(\d+)(st|nd|rd|th)\s+president", claim_lower)
        true_pres = re.search(r"(\d+)(st|nd|rd|th)\s+president", text)

        if claim_pres and true_pres:
            if int(claim_pres.group(1)) != int(true_pres.group(1)):
                return "FALSE", f"Historical records state {true_pres.group(1)}th president."

        claim_years = set(re.findall(r"\d{4}", claim))
        text_years = set(re.findall(r"\d{4}", text))

        if claim_years and text_years and not claim_years.intersection(text_years):
            return "MISLEADING", "Claimed year does not align with verified timelines."

    return None, None

# ===============================
# LLM REASONING
# ===============================
def reason_claim(claim, evidence):
    context = "\n".join([e["content"][:300] for e in evidence])

    prompt = f"""
Explain the factual status of the claim using the evidence.

Claim:
{claim}

Evidence:
{context}

Give a short explanation.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# FACT CHECK PIPELINE
# ===============================
def fact_check_pipeline(claim):
    evidence = retrieve_evidence(claim)

    if not evidence:
        return "UNVERIFIED", "No reliable sources found.", "", ""

    rule_verdict, rule_reason = rule_based_verdict(claim, evidence)

    if rule_verdict:
        verdict = rule_verdict
        explanation = rule_reason
    else:
        verdict = "TRUE"
        explanation = reason_claim(claim, evidence)

    summaries = "\n\n".join(
        [f"{i+1}. {e['title']}\n{e['content'][:250]}..."
         for i, e in enumerate(evidence)]
    )

    links = "\n".join([e["url"] for e in evidence])

    return verdict, explanation, summaries, links

# ===============================
# DEBATE ANALYSIS
# ===============================
def analyze_debate(text):
    sentiment = sentiment_model(text)[0]
    lowered = text.lower()

    bias_orientation = (
        "Positively Biased" if sentiment["label"] == "POSITIVE"
        else "Negatively Biased" if sentiment["label"] == "NEGATIVE"
        else "Neutral"
    )

    fallacies = []
    if "will lead to" in lowered:
        fallacies.append("Slippery Slope")
    if "no alternative" in lowered:
        fallacies.append("False Dilemma")
    if "threat" in lowered or "danger" in lowered:
        fallacies.append("Appeal to Fear")

    return {
        "Sentiment": sentiment,
        "Bias_Orientation": bias_orientation,
        "Logical_Fallacies": fallacies if fallacies else ["None detected"]
    }

# ===============================
# HISTORICAL CLAIM TRAJECTORY
# ===============================
def historical_claim_trajectory(claim):
    queries = [
        claim,
        f"{claim} misinformation",
        f"{claim} political debate"
    ]

    results = []
    for q in queries:
        results.extend(safe_tavily_search(q))

    if not results:
        return "No historical references found.", {}, "LOW"

    timeline, domains = [], []

    for r in results:
        domain = urlparse(r["url"]).netloc
        domains.append(domain)
        timeline.append(f"{r['title']} — {domain}")

    counts = Counter(domains)
    mentions = len(counts)

    risk = "HIGH" if mentions >= 5 else "MEDIUM" if mentions >= 3 else "LOW"

    return "\n".join(sorted(set(timeline))), dict(counts), risk

# ===============================
# GRADIO UI
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Political Intelligence System")

    with gr.Tab("🔍 Fact Check"):
        claim = gr.Textbox(label="Political Claim", lines=3)
        gr.Button("Check Claim").click(
            fact_check_pipeline,
            claim,
            [
                gr.Textbox(label="Verdict"),
                gr.Textbox(label="Explanation", lines=3),
                gr.Textbox(label="Evidence Summary", lines=8),
                gr.Textbox(label="Sources")
            ]
        )

    with gr.Tab("🗣 Debate Analysis"):
        debate = gr.Textbox(label="Debate Text", lines=6)
        gr.Button("Analyze").click(analyze_debate, debate, gr.JSON())

    with gr.Tab("📈 Historical Claim Trajectory"):
        hist_claim = gr.Textbox(label="Political Claim", lines=2)
        gr.Button("Analyze History").click(
            historical_claim_trajectory,
            hist_claim,
            [
                gr.Textbox(label="Timeline", lines=8),
                gr.JSON(label="Source Frequency"),
                gr.Textbox(label="Misinformation Risk Level")
            ]
        )

app.launch(share=True)