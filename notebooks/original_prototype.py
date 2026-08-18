"""
Faithful Python export of the original research Jupyter notebook.
Generated from the uploaded notebook without changing the research logic.
Markdown cells are preserved as comments and code cells use # %% markers.
"""

# %%  # Original notebook code cell 1
#Cell 1: Install Packages

!pip install -q datasets transformers accelerate sentence-transformers rank_bm25 scikit-learn pandas numpy tqdm rouge-score 
!pip install -q seaborn matplotlib bert-score

# %%  # Original notebook code cell 2
#Cell 2: Imports And GPU Check

import os
import re
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, {round(props.total_memory / 1024**3, 2)} GB")

# %%  # Original notebook code cell 3
#Cell 3: Runtime Configuration

# None means evaluate the full split. Set a small number like 30 or 80 only for quick debugging.
MAX_EVAL_EXAMPLES = None

# Retrieval settings
TOP_K = 5
HYBRID_ALPHA = 0.60  # 0.60 = more dense retrieval, 0.40 = more BM25

# Chunking is important for this dataset because some IBM docs are very long
# and may contain noisy embedded/base64 content. Chunk retrieval gives the
# generator the relevant part of the document instead of only the beginning.
CHUNK_WORDS = 360
CHUNK_OVERLAP = 80
PROMPT_CONTEXT_CHAR_BUDGET = 6000

# Generation model.
# For faster runs: "google/flan-t5-base"
# For better quality on T4 x2: "google/flan-t5-large"
GEN_MODEL_NAME = "google/flan-t5-large"
ANSWER_MAX_NEW_TOKENS = 220

# Embedding model.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initial fallback values. Cell 14 tunes these on validation data.
ANSWER_THRESHOLD = 0.60
REQUEST_MORE_EVIDENCE_THRESHOLD = 0.40
MAX_UNSAFE_RATE_FOR_GATE = 0.20

# %% [markdown]  # Original notebook cell 4
# ## Revision notes for dissertation-ready implementation
#
# This version keeps the original answerability-aware RAG design, but makes the evaluation stronger.
#
# Changes made:
#
# 1. End-to-end system evaluation now runs on the full test split by default.
# 2. Ablation study now runs on the full test split by default.
# 3. The notebook includes a manual error analysis table with representative cases.
# 4. Hallucination and unsupported-claim outputs are described as risk/proxy measures, not as human-verified hallucination labels.
# 5. The final demo now separates a cleaner headline demo from a failure-case probe.
# 6. The evaluation table includes retrieval hit information to make wrong-retrieval cases easier to inspect.

# %%  # Original notebook code cell 5
#Cell 4: Load TechQA-RAG-Eval

techqa = load_dataset("nvidia/TechQA-RAG-Eval", split="train")
df = techqa.to_pandas()

print(df.shape)
display(df.head(3))
print(df.columns.tolist())

df["answerable"] = ~df["is_impossible"].astype(bool)

print(df["answerable"].value_counts())

# %%  # Original notebook code cell 6
#Cell 5: Clean Contexts

def normalize_contexts(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

df["contexts"] = df["contexts"].apply(normalize_contexts)

def context_count(ctxs):
    return len(ctxs) if isinstance(ctxs, list) else 0

df["context_count"] = df["contexts"].apply(context_count)

display(df[["id", "question", "answer", "is_impossible", "answerable", "context_count"]].head())
print(df["context_count"].describe())

# %%  # Original notebook code cell 7
#Cell 6: Create Train, Validation, Test Splits

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["answerable"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["answerable"]
)

print("Train:", train_df.shape, train_df["answerable"].value_counts().to_dict())
print("Val:", val_df.shape, val_df["answerable"].value_counts().to_dict())
print("Test:", test_df.shape, test_df["answerable"].value_counts().to_dict())

# %%  # Original notebook code cell 8
#Cell 7: Build Technical Documentation Corpus

def clean_document_text(text):
    text = str(text or "")

    # Remove common embedded image/base64 noise that appears in some long docs.
    text = re.sub(r"data:image/[^ \n\t]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z0-9+/]{250,}={0,2}\b", " ", text)
    text = re.sub(r"(?im)^.*base64.*$", " ", text)

    # Normalize whitespace after noise removal.
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = clean_document_text(text).split()
    if not words:
        return []

    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_words - overlap)

    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break

    return chunks

corpus_rows = []
seen_docs = set()
seen_chunks = set()

for _, row in df.iterrows():
    for ctx in row["contexts"]:
        if not isinstance(ctx, dict):
            continue

        filename = str(ctx.get("filename", "unknown"))
        raw_text = str(ctx.get("text", "")).strip()

        if not raw_text:
            continue

        doc_key = (filename, raw_text[:500])
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)

        doc_id = len(seen_docs) - 1

        for chunk_index, chunk in enumerate(chunk_text(raw_text)):
            chunk_key = (filename, chunk[:500])
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)

            corpus_rows.append({
                "doc_id": doc_id,
                "chunk_id": len(corpus_rows),
                "chunk_index": chunk_index,
                "filename": filename,
                "text": chunk
            })

corpus_df = pd.DataFrame(corpus_rows)

print("Chunked corpus size:", corpus_df.shape)
print("Unique source files:", corpus_df["filename"].nunique())
display(corpus_df.head())

# %% [markdown]  # Original notebook cell 9
# ### Corpus construction note
#
# The retrieval corpus is built from the documentation contexts available in TechQA-RAG-Eval. This is treated as the available technical documentation collection for the RAG system. Train, validation and test splits are still used separately for model training, threshold tuning and final evaluation. Test labels are not used while training the answerability classifier or tuning the gate.
#
# **Honest limitation:** the corpus is built from all rows (including test rows' contexts) rather than from a fixed external documentation set that does not know about test queries. This is a simplification chosen because TechQA-RAG-Eval ships with each question paired with its relevant document(s) rather than against a separately maintained corpus. The practical consequence is that retrieval recall reported here is an upper bound relative to a production setting where the corpus is fixed before test queries arrive. The answerability gate and the gate's decision-level metrics (unsafe answer rate, false abstention rate) are not affected by this in the same way, because the classifier never sees test labels at training time. The proposal acknowledges this explicitly under scope and limitations.

# %%  # Original notebook code cell 10
#Cell 8: Tokenization And BM25 Retriever

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z0-9_./:+#-]+", text)

corpus_texts = corpus_df["text"].tolist()
tokenized_corpus = [tokenize(t) for t in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index ready.")

# %%  # Original notebook code cell 11
#Cell 9: Dense Embeddings

device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

corpus_embeddings = embedder.encode(
    corpus_texts,
    batch_size=96,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Corpus embeddings:", corpus_embeddings.shape)

# %%  # Original notebook code cell 12
#Cell 10: Hybrid Retrieval Function

def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.max() - scores.min() < 1e-8:
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

def retrieve(question, top_k=TOP_K, alpha=HYBRID_ALPHA):
    q_tokens = tokenize(question)
    bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)
    bm25_norm = minmax_normalize(bm25_scores)

    q_emb = embedder.encode(
        [question],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    dense_scores = torch.matmul(q_emb, corpus_embeddings.T).detach().cpu().numpy()[0]
    dense_norm = minmax_normalize(dense_scores)

    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        item = corpus_df.iloc[int(idx)].to_dict()
        item["rank"] = rank
        item["hybrid_score"] = float(hybrid_scores[idx])
        item["dense_score"] = float(dense_scores[idx])
        item["bm25_score"] = float(bm25_scores[idx])
        results.append(item)

    return results

sample_question = df.iloc[0]["question"]
retrieve(sample_question, top_k=3)

# %%  # Original notebook code cell 13
#Cell 11: Retrieval Evaluation

def gold_filenames(row):
    files = []
    for ctx in row["contexts"]:
        if isinstance(ctx, dict) and ctx.get("filename"):
            files.append(str(ctx["filename"]))
    return set(files)

def retrieval_hit(row, top_k=TOP_K):
    if not row["answerable"]:
        return np.nan

    gold = gold_filenames(row)
    if not gold:
        return np.nan

    retrieved = retrieve(row["question"], top_k=top_k)
    retrieved_files = set(r["filename"] for r in retrieved)

    return int(len(gold.intersection(retrieved_files)) > 0)

def evaluate_retrieval(split_df, name):
    hits = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Retrieval {name}"):
        h = retrieval_hit(row, top_k=TOP_K)
        if not pd.isna(h):
            hits.append(h)

    recall = np.mean(hits) if hits else np.nan
    print(f"{name} retrieval recall@{TOP_K}: {recall:.4f}")

evaluate_retrieval(val_df, "Validation")
evaluate_retrieval(test_df, "Test")

# %%  # Original notebook code cell 14
#Cell 12: Answerability Feature Extraction

def lexical_overlap_ratio(question, context):
    q = set(tokenize(question))
    c = set(tokenize(context))
    if not q:
        return 0.0
    return len(q.intersection(c)) / len(q)

def make_answerability_features(row, top_k=TOP_K):
    retrieved = retrieve(row["question"], top_k=top_k)

    scores = [r["hybrid_score"] for r in retrieved]
    dense_scores = [r["dense_score"] for r in retrieved]
    bm25_scores = [r["bm25_score"] for r in retrieved]
    overlaps = [lexical_overlap_ratio(row["question"], r["text"]) for r in retrieved]

    joined_context = "\n".join(r["text"] for r in retrieved)

    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0

    return {
        "top_hybrid": top_score,
        "mean_hybrid": float(np.mean(scores)) if scores else 0.0,
        "score_gap": top_score - second_score,
        "score_gap_ratio": float((top_score - second_score) / (abs(top_score) + 1e-6)),
        "top_dense": dense_scores[0] if dense_scores else 0.0,
        "mean_dense": float(np.mean(dense_scores)) if dense_scores else 0.0,
        "top_bm25": bm25_scores[0] if bm25_scores else 0.0,
        "mean_bm25": float(np.mean(bm25_scores)) if bm25_scores else 0.0,
        "top_lexical_overlap": overlaps[0] if overlaps else 0.0,
        "mean_lexical_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "lexical_overlap": lexical_overlap_ratio(row["question"], joined_context),
        "question_len": len(str(row["question"]).split()),
        "retrieved_chars": len(joined_context),
    }

def build_feature_table(split_df):
    features = []
    labels = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        features.append(make_answerability_features(row))
        labels.append(int(row["answerable"]))

    return pd.DataFrame(features), np.array(labels)

X_train, y_train = build_feature_table(train_df)
X_val, y_val = build_feature_table(val_df)
X_test, y_test = build_feature_table(test_df)

display(X_train.head())

# %%  # Original notebook code cell 15
#Cell 13: Train Answerability Classifier

from sklearn.isotonic import IsotonicRegression

FEATURE_COLUMNS = X_train.columns.tolist()

answerability_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

answerability_clf.fit(X_train[FEATURE_COLUMNS], y_train)

raw_val_prob_for_calibration = answerability_clf.predict_proba(X_val[FEATURE_COLUMNS])[:, 1]
answerability_calibrator = IsotonicRegression(out_of_bounds="clip")
answerability_calibrator.fit(raw_val_prob_for_calibration, y_val)

def calibrated_answerability_prob(X):
    raw_prob = answerability_clf.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    cal_prob = answerability_calibrator.transform(raw_prob)
    return raw_prob, cal_prob

def evaluate_answerability(X, y, name):
    raw_prob, prob = calibrated_answerability_prob(X)
    pred = (prob >= 0.50).astype(int)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y, pred))

    try:
        print("AUROC:", roc_auc_score(y, prob))
    except Exception:
        pass

    print(classification_report(y, pred, target_names=["unanswerable", "answerable"]))
    print(confusion_matrix(y, pred))

evaluate_answerability(X_val, y_val, "Validation")
evaluate_answerability(X_test, y_test, "Test")

# %%  # Original notebook code cell 16
# Cell 13b — Feature Importance For Answerability Classifier
# This figure directly addresses RQ3 ("Which retrieval and question based features
# are most useful for predicting answerability?"). The Random Forest exposes a
# Gini-based importance per feature; this is a coarse but standard signal.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

feature_importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": answerability_clf.feature_importances_
}).sort_values("importance", ascending=True).reset_index(drop=True)

print("Feature importance ranking:")
display(feature_importance_df.iloc[::-1].round(4))

plt.figure(figsize=(9, 6))
sns.barplot(
    data=feature_importance_df,
    x="importance",
    y="feature",
    color="#4c78a8"
)
plt.title("Random Forest Feature Importance (Answerability Classifier)")
plt.xlabel("Importance (Gini)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# %%  # Original notebook code cell 17
#Cell 14: Decision Logic

def tune_answer_threshold(y_true, prob_answerable, max_unsafe_rate=MAX_UNSAFE_RATE_FOR_GATE):
    candidates = np.linspace(0.20, 0.85, 66)
    best = None

    for threshold in candidates:
        pred_answer = prob_answerable >= threshold

        unanswerable_mask = y_true == 0
        answerable_mask = y_true == 1

        unsafe_rate = (
            pred_answer[unanswerable_mask].mean()
            if unanswerable_mask.any() else 0.0
        )
        false_abstention_rate = (
            (~pred_answer[answerable_mask]).mean()
            if answerable_mask.any() else 0.0
        )

        accuracy = (pred_answer.astype(int) == y_true).mean()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            pred_answer.astype(int),
            average="binary",
            zero_division=0
        )

        allowed = unsafe_rate <= max_unsafe_rate
        score = f1 - 0.30 * false_abstention_rate - 0.70 * unsafe_rate

        row = {
            "threshold": float(threshold),
            "unsafe_rate": float(unsafe_rate),
            "false_abstention_rate": float(false_abstention_rate),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "allowed": allowed,
            "score": float(score)
        }

        if best is None:
            best = row
        elif allowed and (not best["allowed"] or row["score"] > best["score"]):
            best = row
        elif not best["allowed"] and row["unsafe_rate"] < best["unsafe_rate"]:
            best = row

    return best

_, val_prob_calibrated = calibrated_answerability_prob(X_val)
threshold_info = tune_answer_threshold(y_val, val_prob_calibrated)

ANSWER_THRESHOLD = threshold_info["threshold"]
REQUEST_MORE_EVIDENCE_THRESHOLD = max(0.05, ANSWER_THRESHOLD - 0.20)

print("Tuned ANSWER_THRESHOLD:", round(ANSWER_THRESHOLD, 3))
print("REQUEST_MORE_EVIDENCE_THRESHOLD:", round(REQUEST_MORE_EVIDENCE_THRESHOLD, 3))
print("Validation threshold metrics:", threshold_info)

def predict_answerability(question):
    fake_row = {
        "question": question,
        "answerable": True,
        "contexts": []
    }

    features = pd.DataFrame([make_answerability_features(fake_row)])
    features = features.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

    raw_prob = answerability_clf.predict_proba(features)[0, 1]
    prob_answerable = answerability_calibrator.transform([raw_prob])[0]

    feature_dict = features.iloc[0].to_dict()
    feature_dict["raw_prob_answerable"] = float(raw_prob)
    feature_dict["calibrated_prob_answerable"] = float(prob_answerable)

    return float(prob_answerable), feature_dict

def decide_action(prob_answerable):
    if prob_answerable >= ANSWER_THRESHOLD:
        return "answer"
    elif prob_answerable >= REQUEST_MORE_EVIDENCE_THRESHOLD:
        return "request_more_evidence"
    else:
        return "abstain"

q = df.iloc[0]["question"]
p, feats = predict_answerability(q)
print("Question:", q)
print("P(answerable):", p)
print("Decision:", decide_action(p))

# %%  # Original notebook code cell 18
# Cell 14b — Answerability Probability Distribution By True Label
# Shows whether the gate's calibrated probability actually separates answerable
# from unanswerable questions on the test split. Vertical lines mark the tuned
# thresholds, so the three decision regions (answer / request / abstain) are
# visible against the empirical distribution.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

_, test_prob_for_dist = calibrated_answerability_prob(X_test)

dist_df = pd.DataFrame({
    "prob_answerable": test_prob_for_dist,
    "true_label": ["answerable" if y else "unanswerable" for y in y_test]
})

plt.figure(figsize=(10, 6))
sns.histplot(
    data=dist_df,
    x="prob_answerable",
    hue="true_label",
    bins=25,
    kde=True,
    common_norm=False,
    stat="density",
    palette={"answerable": "#2a9d8f", "unanswerable": "#e76f51"},
    alpha=0.55
)

plt.axvline(
    ANSWER_THRESHOLD,
    color="black", linestyle="--", linewidth=1.5,
    label=f"Answer threshold ({ANSWER_THRESHOLD:.2f})"
)
plt.axvline(
    REQUEST_MORE_EVIDENCE_THRESHOLD,
    color="gray", linestyle=":", linewidth=1.5,
    label=f"Request-more-evidence threshold ({REQUEST_MORE_EVIDENCE_THRESHOLD:.2f})"
)

plt.title("Calibrated Answerability Probability by True Label (Test Split)")
plt.xlabel("Calibrated P(answerable)")
plt.ylabel("Density")
plt.legend(loc="upper center")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# Decision-region counts
abstain_mask = test_prob_for_dist < REQUEST_MORE_EVIDENCE_THRESHOLD
request_mask = (test_prob_for_dist >= REQUEST_MORE_EVIDENCE_THRESHOLD) & (test_prob_for_dist < ANSWER_THRESHOLD)
answer_mask = test_prob_for_dist >= ANSWER_THRESHOLD

region_counts = pd.DataFrame({
    "region": ["abstain", "request_more_evidence", "answer"],
    "answerable_count": [
        int(((y_test == 1) & abstain_mask).sum()),
        int(((y_test == 1) & request_mask).sum()),
        int(((y_test == 1) & answer_mask).sum()),
    ],
    "unanswerable_count": [
        int(((y_test == 0) & abstain_mask).sum()),
        int(((y_test == 0) & request_mask).sum()),
        int(((y_test == 0) & answer_mask).sum()),
    ],
})
print("Decision-region distribution on test split:")
display(region_counts)

# %%  # Original notebook code cell 19
#Cell 15: Load Generator Model

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)

model_kwargs = {}

if torch.cuda.is_available():
    model_kwargs["torch_dtype"] = torch.float16
    model_kwargs["device_map"] = "auto"

gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    GEN_MODEL_NAME,
    **model_kwargs
)

if not torch.cuda.is_available():
    gen_model = gen_model.to("cpu")

gen_model.eval()

print("Generator loaded:", GEN_MODEL_NAME)

# %%  # Original notebook code cell 20
#Cell 16: RAG Prompt And Answer Generation

REFUSAL_TEXT = "Unable to answer based on the provided documentation."

def compact_context_text(text, max_chars=1600):
    text = clean_document_text(text)
    return text[:max_chars].strip()

def build_rag_prompt(question, retrieved):
    context_blocks = []
    used_chars = 0

    for r in retrieved:
        text = compact_context_text(r["text"], max_chars=1800)
        if not text:
            continue

        block = (
            f"[Document rank {r.get('rank', '?')}: {r['filename']} "
            f"| score={r.get('hybrid_score', 0.0):.3f}]\n{text}"
        )

        if used_chars + len(block) > PROMPT_CONTEXT_CHAR_BUDGET:
            remaining = PROMPT_CONTEXT_CHAR_BUDGET - used_chars
            if remaining <= 300:
                break
            block = block[:remaining]

        context_blocks.append(block)
        used_chars += len(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a technical documentation assistant.

Use only the provided documentation excerpts.

Rules:
- If the excerpts contain the answer, give a concise technical answer.
- Do not say you are unable to answer when the answer is present in the excerpts.
- If the excerpts truly do not contain enough information, reply exactly:
"{REFUSAL_TEXT}"
- Do not invent details that are not supported by the excerpts.

Question:
{question}

Documentation excerpts:
{context}

Answer:
""".strip()

    return prompt

def clean_generated_answer(answer, prompt=None):
    answer = str(answer or "").strip()

    if prompt and answer.startswith(prompt):
        answer = answer[len(prompt):].strip()

    answer = re.sub(r"^\s*Answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer

@torch.inference_mode()
def generate_answer(question, retrieved, max_new_tokens=ANSWER_MAX_NEW_TOKENS):
    prompt = build_rag_prompt(question, retrieved)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    first_device = next(gen_model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.08,
        length_penalty=0.90
    )

    if getattr(gen_model.config, "is_encoder_decoder", False):
        answer_ids = outputs[0]
    else:
        input_len = inputs["input_ids"].shape[-1]
        answer_ids = outputs[0][input_len:]

    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return clean_generated_answer(answer, prompt=prompt)

# %%  # Original notebook code cell 21
#Cell 17: Claim-Level Support Proxy

def split_sentences(text):
    text = str(text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]

def chunk_context(text, max_chars=700):
    sentences = split_sentences(text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks[:30]

def claim_support_analysis(answer, retrieved, threshold=0.42):
    claims = split_sentences(answer)

    context_text = "\n".join(r["text"] for r in retrieved)
    context_chunks = chunk_context(context_text)

    if not claims:
        return {
            "num_claims": 0,
            "unsupported_claims": [],
            "unsupported_rate": 0.0,
            "claim_scores": []
        }

    if not context_chunks:
        return {
            "num_claims": len(claims),
            "unsupported_claims": claims,
            "unsupported_rate": 1.0,
            "claim_scores": [0.0] * len(claims)
        }

    claim_emb = embedder.encode(
        claims,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    ctx_emb = embedder.encode(
        context_chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    sim = torch.matmul(claim_emb, ctx_emb.T).detach().cpu().numpy()
    max_scores = sim.max(axis=1)

    unsupported = [
        claim for claim, score in zip(claims, max_scores)
        if score < threshold
    ]

    return {
        "num_claims": len(claims),
        "unsupported_claims": unsupported,
        "unsupported_rate": len(unsupported) / len(claims),
        "claim_scores": max_scores.tolist()
    }

# %% [markdown]  # Original notebook cell 22
# ### Interpretation note on hallucination and support metrics
#
# The unsupported claim rate in this notebook is a similarity-based support proxy. It helps flag answers whose claims do not appear well supported by retrieved documentation chunks. It should not be reported as a human-verified hallucination label. RAGTruth is used as a supporting hallucination-analysis resource, while TechQA-RAG-Eval remains the main technical documentation QA dataset.

# %%  # Original notebook code cell 23
#Cell 18: Full System Prediction (with real request_more_evidence escalation)

# When the gate falls in the middle band, the system now expands retrieval
# from top-K to top-K * 2, recomputes the answerability features against the
# wider context, and re-scores the gate. If the second pass scores above the
# answer threshold the system commits to generation against the expanded
# context. If it still falls in the middle band or below, the system abstains.
# This makes "request_more_evidence" a real action in offline evaluation
# rather than a synonym for abstention.

EXPANDED_TOP_K = TOP_K * 2  # top 10 by default

def make_features_from_retrieved(question, retrieved):
    """Compute the same feature vector as make_answerability_features but
    against an already-retrieved set of chunks (used for the second-pass
    escalation so we don't double-retrieve)."""
    scores = [r["hybrid_score"] for r in retrieved]
    dense_scores = [r["dense_score"] for r in retrieved]
    bm25_scores = [r["bm25_score"] for r in retrieved]
    overlaps = [lexical_overlap_ratio(question, r["text"]) for r in retrieved]
    joined_context = "\n".join(r["text"] for r in retrieved)
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    return {
        "top_hybrid": top_score,
        "mean_hybrid": float(np.mean(scores)) if scores else 0.0,
        "score_gap": top_score - second_score,
        "score_gap_ratio": float((top_score - second_score) / (abs(top_score) + 1e-6)),
        "top_dense": dense_scores[0] if dense_scores else 0.0,
        "mean_dense": float(np.mean(dense_scores)) if dense_scores else 0.0,
        "top_bm25": bm25_scores[0] if bm25_scores else 0.0,
        "mean_bm25": float(np.mean(bm25_scores)) if bm25_scores else 0.0,
        "top_lexical_overlap": overlaps[0] if overlaps else 0.0,
        "mean_lexical_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "lexical_overlap": lexical_overlap_ratio(question, joined_context),
        "question_len": len(str(question).split()),
        "retrieved_chars": len(joined_context),
    }

def score_features(feature_dict):
    """Run the calibrated answerability gate on a single feature dict."""
    features = pd.DataFrame([feature_dict])
    features = features.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    raw_prob = answerability_clf.predict_proba(features)[0, 1]
    prob = answerability_calibrator.transform([raw_prob])[0]
    return float(prob), float(raw_prob)

def is_refusal_answer(answer):
    text = re.sub(r"\s+", " ", str(answer or "").lower()).strip()
    refusal_phrases = [
        "unable to answer",
        "not enough information",
        "insufficient information",
        "provided documentation does not",
        "provided context does not",
    ]
    return any(p in text for p in refusal_phrases)

def sentence_split(text):
    text = clean_document_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 6]

def extractive_context_answer(question, retrieved, max_sentences=4):
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return ""

    candidates = []

    for r in retrieved:
        rank_bonus = 1.0 / max(1, int(r.get("rank", 1)))

        for sent in sentence_split(r["text"]):
            s_tokens = set(tokenize(sent))
            if not s_tokens:
                continue

            overlap = len(q_tokens.intersection(s_tokens))
            overlap_ratio = overlap / max(1, len(q_tokens))

            detail_bonus = min(len(sent.split()) / 40.0, 1.0)
            score = overlap_ratio + 0.15 * detail_bonus + 0.10 * rank_bonus

            if overlap > 0:
                candidates.append({
                    "score": score,
                    "rank": int(r.get("rank", 1)),
                    "sentence": sent
                })

    if not candidates:
        return ""

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected = []
    seen = set()

    for c in candidates:
        normalized = re.sub(r"\W+", " ", c["sentence"].lower()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(c["sentence"])
        if len(selected) >= max_sentences:
            break

    return " ".join(selected).strip()

def answerability_aware_rag(question, top_k=TOP_K):
    # First pass at top_k.
    retrieved = retrieve(question, top_k=top_k)
    features_first = make_features_from_retrieved(question, retrieved)
    prob_answerable, raw_prob = score_features(features_first)
    escalated = False

    # Initial decision based on first-pass probability.
    if prob_answerable >= ANSWER_THRESHOLD:
        decision = "answer"
    elif prob_answerable >= REQUEST_MORE_EVIDENCE_THRESHOLD:
        decision = "request_more_evidence"
    else:
        decision = "abstain"

    # If the gate landed in the middle band, escalate retrieval and re-score.
    # The second-pass result either upgrades to "answer" against the wider
    # context, or falls back to "abstain".
    if decision == "request_more_evidence":
        escalated = True
        retrieved_expanded = retrieve(question, top_k=EXPANDED_TOP_K)
        features_second = make_features_from_retrieved(question, retrieved_expanded)
        prob_answerable_second, raw_prob_second = score_features(features_second)

        if prob_answerable_second >= ANSWER_THRESHOLD:
            decision = "answer"
            retrieved = retrieved_expanded
            prob_answerable = prob_answerable_second
            raw_prob = raw_prob_second
            features_first = features_second
        else:
            decision = "abstain"

    answer_source = "policy"

    if decision == "abstain":
        answer = REFUSAL_TEXT
        support = {"num_claims": 0, "unsupported_claims": [], "unsupported_rate": 0.0, "claim_scores": []}

    else:  # decision == "answer"
        answer = generate_answer(question, retrieved)
        answer_source = "generated"

        if is_refusal_answer(answer) or len(str(answer).split()) < 4:
            fallback = extractive_context_answer(question, retrieved)
            if fallback:
                answer = fallback
                answer_source = "extractive_fallback"
            else:
                decision = "abstain"
                answer = REFUSAL_TEXT
                answer_source = "fallback_no_supported_sentence"

        support = (
            claim_support_analysis(answer, retrieved)
            if decision == "answer"
            else {"num_claims": 0, "unsupported_claims": [], "unsupported_rate": 0.0, "claim_scores": []}
        )

    features_first["raw_prob_answerable"] = raw_prob
    features_first["calibrated_prob_answerable"] = prob_answerable

    return {
        "question": question,
        "decision": decision,
        "prob_answerable": prob_answerable,
        "raw_prob_answerable": raw_prob,
        "answer": answer,
        "answer_source": answer_source,
        "retrieved": retrieved,
        "support": support,
        "features": features_first,
        "escalated": escalated,
    }

example = answerability_aware_rag(df.iloc[0]["question"])
print("Decision:", example["decision"])
print("P(answerable):", round(example["prob_answerable"], 3))
print("Raw P(answerable):", round(example["raw_prob_answerable"], 3))
print("Answer source:", example["answer_source"])
print("Escalated:", example["escalated"])
print("Answer:", example["answer"])
print("Unsupported rate:", example["support"]["unsupported_rate"])

# %%  # Original notebook code cell 24
#Cell 19: Evaluate End-To-End System

# This is the main system-level evaluation for the dissertation.
# By default it runs on the full test split because MAX_EVAL_EXAMPLES = None.

def rouge_l(prediction, reference):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(str(reference), str(prediction))["rougeL"].fmeasure

def evaluate_system(split_df, name="test", max_examples=MAX_EVAL_EXAMPLES):
    if max_examples is None:
        sample_df = split_df.reset_index(drop=True).copy()
        sample_note = f"full split, {len(sample_df)} rows"
    else:
        sample_df = split_df.sample(
            n=min(max_examples, len(split_df)),
            random_state=SEED
        ).reset_index(drop=True)
        sample_note = f"sample, {len(sample_df)} rows"

    print(f"Running {name} evaluation on {sample_note}.")

    rows = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"System eval {name}"):
        out = answerability_aware_rag(row["question"], top_k=TOP_K)

        system_answered = out["decision"] == "answer"
        true_answerable = bool(row["answerable"])
        generated_is_refusal = is_refusal_answer(out["answer"])

        r_l = np.nan
        if true_answerable and system_answered and not generated_is_refusal:
            r_l = rouge_l(out["answer"], row["answer"])

        support = out.get("support", {})
        retrieved_files = [r["filename"] for r in out["retrieved"]]
        gold_files = sorted(gold_filenames(row))

        retrieval_hit_value = np.nan
        if true_answerable and gold_files:
            retrieval_hit_value = int(len(set(gold_files).intersection(set(retrieved_files))) > 0)

        top_retrieved_score = np.nan
        if out["retrieved"]:
            top_retrieved_score = out["retrieved"][0].get("hybrid_score", np.nan)

        rows.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["answer"],
            "true_answerable": true_answerable,
            "decision": out["decision"],
            "prob_answerable": out["prob_answerable"],
            "raw_prob_answerable": out.get("raw_prob_answerable", np.nan),
            "answer_source": out.get("answer_source", "unknown"),
            "generated_answer": out["answer"] if system_answered and not generated_is_refusal else np.nan,
            "non_answer_reason": out["answer"] if not system_answered or generated_is_refusal else np.nan,
            "rougeL": r_l,
            "unsupported_claim_rate": support.get("unsupported_rate", np.nan) if system_answered else np.nan,
            "num_claims": support.get("num_claims", 0) if system_answered else 0,
            "retrieved_filenames": retrieved_files,
            "gold_filenames": gold_files,
            "retrieval_hit": retrieval_hit_value,
            "top_retrieved_filename": retrieved_files[0] if retrieved_files else np.nan,
            "top_retrieved_score": top_retrieved_score,
        })

    results = pd.DataFrame(rows)

    true = results["true_answerable"].astype(int)
    pred_answer = (results["decision"] == "answer").astype(int)

    print(f"\n{name.upper()} DECISION METRICS")
    print(classification_report(true, pred_answer, target_names=["should_not_answer", "should_answer"]))

    unanswerable = results[results["true_answerable"] == False]
    answerable = results[results["true_answerable"] == True]

    unsafe_answer_rate = (
        (unanswerable["decision"] == "answer").mean()
        if len(unanswerable) else np.nan
    )

    false_abstention_rate = (
        (answerable["decision"] != "answer").mean()
        if len(answerable) else np.nan
    )

    retrieval_hit_rate = results.loc[
        results["retrieval_hit"].notna(), "retrieval_hit"
    ].mean()

    system_eval_summary = {
        "split": name,
        "rows_evaluated": len(results),
        "decision_accuracy": accuracy_score(true, pred_answer),
        "unsafe_answer_rate": unsafe_answer_rate,
        "false_abstention_rate": false_abstention_rate,
        "retrieval_hit_rate_on_answerable": retrieval_hit_rate,
        "mean_rougeL_answered_answerable": results["rougeL"].mean(),
        "mean_unsupported_claim_rate_answered": results["unsupported_claim_rate"].mean(),
    }

    print("Unsafe answer rate on unanswerable questions:", unsafe_answer_rate)
    print("False abstention/request rate on answerable questions:", false_abstention_rate)
    print("Retrieval hit rate on answerable questions:", retrieval_hit_rate)
    print("Mean ROUGE-L for answered answerable questions:", results["rougeL"].mean())
    print("Mean unsupported claim rate for answered questions:", results["unsupported_claim_rate"].mean())

    display(pd.DataFrame([system_eval_summary]).round(4))

    return results

test_results = evaluate_system(test_df, name="test", max_examples=MAX_EVAL_EXAMPLES)
display(test_results[[
    "id",
    "question",
    "gold_answer",
    "true_answerable",
    "decision",
    "prob_answerable",
    "answer_source",
    "retrieval_hit",
    "top_retrieved_filename",
    "generated_answer",
    "non_answer_reason",
    "rougeL",
    "unsupported_claim_rate"
]].head(10))

# %%  # Original notebook code cell 25
#Cell 20: Inspect Errors

unsafe_answers = test_results[
    (test_results["true_answerable"] == False) &
    (test_results["decision"] == "answer")
]

false_abstentions = test_results[
    (test_results["true_answerable"] == True) &
    (test_results["decision"] != "answer")
]

low_grounding = test_results[
    (test_results["decision"] == "answer") &
    (test_results["unsupported_claim_rate"] > 0.5)
]

wrong_retrieval_candidates = test_results[
    (test_results["true_answerable"] == True) &
    (test_results["decision"] == "answer") &
    (test_results["retrieval_hit"] == 0)
]

print("Unsafe answers:", len(unsafe_answers))
display(unsafe_answers[[
    "id",
    "question",
    "decision",
    "prob_answerable",
    "answer_source",
    "top_retrieved_filename",
    "generated_answer"
]].head())

print("False abstentions / requests:", len(false_abstentions))
display(false_abstentions[[
    "id",
    "question",
    "gold_answer",
    "decision",
    "prob_answerable",
    "non_answer_reason"
]].head())

print("Low grounding answers:", len(low_grounding))
display(low_grounding[[
    "id",
    "question",
    "decision",
    "answer_source",
    "unsupported_claim_rate",
    "generated_answer"
]].head())

print("Wrong retrieval candidates:", len(wrong_retrieval_candidates))
display(wrong_retrieval_candidates[[
    "id",
    "question",
    "gold_filenames",
    "retrieved_filenames",
    "top_retrieved_filename",
    "generated_answer"
]].head())

# %%  # Original notebook code cell 26
#Cell 20b: Manual Error Analysis Sample

# This table is meant for human inspection in the dissertation write-up.
# Fill manual_note after reading the question, retrieved files, generated answer and gold answer.

def categorize_for_manual_review(row):
    if (row["true_answerable"] == False) and (row["decision"] == "answer"):
        return "Unsafe answer"
    if (row["true_answerable"] == True) and (row["decision"] != "answer"):
        return "False abstention or request"
    if row["decision"] == "request_more_evidence":
        return "Request more evidence"
    if (row["decision"] == "answer") and pd.notna(row["unsupported_claim_rate"]) and row["unsupported_claim_rate"] > 0.5:
        return "Unsupported generated answer"
    if (row["true_answerable"] == True) and (row["decision"] == "answer") and (row["retrieval_hit"] == 0):
        return "Wrong retrieval likely"
    if (row["true_answerable"] == True) and (row["decision"] == "answer"):
        return "Good answer candidate"
    return "Other"

manual_review_df = test_results.copy()
manual_review_df["case_type"] = manual_review_df.apply(categorize_for_manual_review, axis=1)
manual_review_df["manual_note"] = ""

# Take a balanced sample: up to 3 examples per category, maximum 15 rows.
case_priority = [
    "Good answer candidate",
    "Unsafe answer",
    "False abstention or request",
    "Request more evidence",
    "Unsupported generated answer",
    "Wrong retrieval likely",
    "Other",
]

manual_sample_parts = []
for case in case_priority:
    part = manual_review_df[manual_review_df["case_type"] == case].head(3)
    if len(part):
        manual_sample_parts.append(part)

manual_error_analysis_sample = (
    pd.concat(manual_sample_parts, ignore_index=True)
    .head(15)
    if manual_sample_parts else pd.DataFrame()
)

manual_cols = [
    "case_type",
    "id",
    "question",
    "true_answerable",
    "decision",
    "prob_answerable",
    "retrieval_hit",
    "unsupported_claim_rate",
    "gold_answer",
    "generated_answer",
    "non_answer_reason",
    "gold_filenames",
    "retrieved_filenames",
    "manual_note",
]

display(manual_error_analysis_sample[manual_cols])

print("Case type counts:")
display(manual_review_df["case_type"].value_counts().rename_axis("case_type").reset_index(name="count"))

# %%  # Original notebook code cell 27
#Cell 21: Optional RAGTruth Load For Hallucination Analysis

RAGTRUTH_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"

try:
    ragtruth_responses = pd.read_json(
        f"{RAGTRUTH_BASE}/response.jsonl",
        lines=True
    )

    ragtruth_sources = pd.read_json(
        f"{RAGTRUTH_BASE}/source_info.jsonl",
        lines=True
    )

    ragtruth = ragtruth_responses.merge(
        ragtruth_sources,
        on="source_id",
        how="left"
    )

    ragtruth["has_hallucination"] = ragtruth["labels"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )

    ragtruth_qa = ragtruth[ragtruth["task_type"] == "QA"].copy()

    print("RAGTruth responses:", ragtruth.shape)
    print("RAGTruth QA subset:", ragtruth_qa.shape)
    print(ragtruth_qa["has_hallucination"].value_counts(normalize=True))

    display(ragtruth_qa.head(3))

except Exception as e:
    print("Could not load RAGTruth directly. Check Kaggle internet setting.")
    print(e)
    ragtruth_qa = pd.DataFrame()

# %%  # Original notebook code cell 28
#Cell 22: Optional RAGTruth Label-Type Summary

if len(ragtruth_qa):
    label_types = []

    for labels in ragtruth_qa["labels"]:
        if isinstance(labels, list):
            for lab in labels:
                if isinstance(lab, dict):
                    label_types.append(lab.get("label_type", "unknown"))

    label_type_counts = pd.Series(label_types).value_counts()
    display(label_type_counts.head(20))

# %%  # Original notebook code cell 29
#Cell 23: Optional Hallucination-Risk Proxy Model From RAGTruth

def source_info_to_text(source_info):
    if isinstance(source_info, dict):
        if "passages" in source_info:
            return str(source_info["passages"])
        return json.dumps(source_info)
    return str(source_info)

def hallucination_features(context, response):
    context = str(context)
    response = str(response)

    c_tokens = set(tokenize(context))
    r_tokens = tokenize(response)
    r_set = set(r_tokens)

    overlap = len(c_tokens.intersection(r_set)) / max(1, len(r_set))

    return {
        "response_len": len(response.split()),
        "context_len": len(context.split()),
        "token_overlap": overlap,
        "num_sentences": len(split_sentences(response)),
        "num_digits": sum(ch.isdigit() for ch in response),
    }

if len(ragtruth_qa):
    ragtruth_qa["context_text"] = ragtruth_qa["source_info"].apply(source_info_to_text)

    h_features = []
    h_labels = []

    for _, row in tqdm(ragtruth_qa.iterrows(), total=len(ragtruth_qa)):
        h_features.append(hallucination_features(row["context_text"], row["response"]))
        h_labels.append(int(row["has_hallucination"]))

    H = pd.DataFrame(h_features)
    y_h = np.array(h_labels)

    H_train, H_test, yh_train, yh_test = train_test_split(
        H,
        y_h,
        test_size=0.25,
        random_state=SEED,
        stratify=y_h
    )

    hallucination_clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    hallucination_clf.fit(H_train, yh_train)

    yh_pred = hallucination_clf.predict(H_test)
    yh_prob = hallucination_clf.predict_proba(H_test)[:, 1]

    print(classification_report(yh_test, yh_pred, target_names=["no_hallucination", "hallucination"]))

    try:
        print("AUROC:", roc_auc_score(yh_test, yh_prob))
    except Exception:
        pass
else:
    hallucination_clf = None
    print("RAGTruth model skipped.")

# %%  # Original notebook code cell 30
# Cell 23b — RAGTruth Label Integration + Visualization

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

sns.set(style="whitegrid")

if "ragtruth_qa" not in globals() or len(ragtruth_qa) == 0:
    print("ragtruth_qa is empty. Run the RAGTruth loading cell first.")
else:
    rt = ragtruth_qa.copy()

    # Ensure binary hallucination flag exists
    if "has_hallucination" not in rt.columns:
        rt["has_hallucination"] = rt["labels"].apply(lambda x: isinstance(x, list) and len(x) > 0)

    # Extract label types from span labels
    def extract_label_types(labels):
        out = []
        if isinstance(labels, list):
            for lab in labels:
                if isinstance(lab, dict):
                    out.append(lab.get("label_type", "unknown"))
        return out

    def dominant_type(types):
        if not types:
            return "No Hallucination"
        return Counter(types).most_common(1)[0][0]

    rt["label_types"] = rt["labels"].apply(extract_label_types)
    rt["num_labels"] = rt["label_types"].apply(len)
    rt["dominant_label_type"] = rt["label_types"].apply(dominant_type)

    # Integrate hallucination-risk model predictions (if model is available)
    if "hallucination_clf" in globals() and hallucination_clf is not None:
        if "context_text" not in rt.columns:
            rt["context_text"] = rt["source_info"].apply(source_info_to_text)

        H_rt = pd.DataFrame([
            hallucination_features(c, r)
            for c, r in zip(rt["context_text"], rt["response"])
        ])
        rt["pred_hallucination_risk"] = hallucination_clf.predict_proba(H_rt)[:, 1]
    else:
        rt["pred_hallucination_risk"] = np.nan

    # Exploded frame for label-frequency view
    exp = rt[["has_hallucination", "label_types", "pred_hallucination_risk"]].explode("label_types")
    exp["label_types"] = exp["label_types"].fillna("No Hallucination")

    # ---- Metrics summary ----
    print("RAGTruth QA rows:", len(rt))
    print("Hallucination prevalence:", round(float(rt["has_hallucination"].mean()), 4))
    print("\nTop label types:")
    display(exp["label_types"].value_counts().head(10))

    # ---- Visualization 1: Label-type frequency ----
    plt.figure(figsize=(10, 6))
    top_types = exp["label_types"].value_counts().head(10).index
    sns.countplot(
        data=exp[exp["label_types"].isin(top_types)],
        y="label_types",
        order=top_types,
        color="#4c78a8"
    )
    plt.title("RAGTruth Label-Type Frequency (Top 10)")
    plt.xlabel("Count")
    plt.ylabel("Label type")
    plt.tight_layout()
    plt.show()

    # ---- Visualization 2: Hallucination prevalence by dominant label type ----
    plt.figure(figsize=(10, 6))
    dom_prev = (
        rt.groupby("dominant_label_type")["has_hallucination"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    sns.barplot(data=dom_prev, x="has_hallucination", y="dominant_label_type", color="#f58518")
    plt.title("Hallucination Rate by Dominant Label Type")
    plt.xlabel("Rate")
    plt.ylabel("Dominant label type")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

    # ---- Visualization 3: Predicted risk by true hallucination flag ----
    if rt["pred_hallucination_risk"].notna().any():
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(
            data=rt,
            x="has_hallucination",
            y="pred_hallucination_risk"
        )
        ax.set_xticklabels(["No Hallucination", "Hallucination"])
        plt.title("Predicted Hallucination Risk vs True Flag")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Notice] hallucination_clf not available: Skipping 'Predicted Risk vs True Flag' plot.")

    # ---- Visualization 4: Predicted risk by dominant label type ----
    if rt["pred_hallucination_risk"].notna().any():
        plt.figure(figsize=(10, 6))
        risk_by_type = (
            rt.groupby("dominant_label_type")["pred_hallucination_risk"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        sns.barplot(
            data=risk_by_type,
            x="pred_hallucination_risk",
            y="dominant_label_type",
            color="#54a24b"
        )
        plt.title("Mean Predicted Risk by Dominant Label Type")
        plt.xlabel("Mean predicted risk")
        plt.ylabel("Dominant label type")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Notice] hallucination_clf not available: Skipping 'Mean Predicted Risk by Type' plot.")

# %%  # Original notebook code cell 31
#Cell 24: Add Hallucination Risk Proxy To System Outputs

def estimate_hallucination_risk(answer, retrieved):
    if hallucination_clf is None:
        return np.nan

    context = "\n".join(r["text"] for r in retrieved)
    feats = pd.DataFrame([hallucination_features(context, answer)])
    risk = hallucination_clf.predict_proba(feats)[0, 1]
    return float(risk)

def answerability_aware_rag_with_risk(question, top_k=TOP_K):
    out = answerability_aware_rag(question, top_k=top_k)
    out["hallucination_risk_proxy"] = estimate_hallucination_risk(
        out["answer"],
        out["retrieved"]
    )
    return out

example_risk = answerability_aware_rag_with_risk(df.iloc[1]["question"])

print("Decision:", example_risk["decision"])
print("P(answerable):", round(example_risk["prob_answerable"], 3))
print("Hallucination risk proxy:", example_risk["hallucination_risk_proxy"])
print("Answer:", example_risk["answer"])

# %%  # Original notebook code cell 32
#Cell 25: Final Demo And Failure Case Probe

# Use this cell for presentation. It shows one cleaner demo from the evaluated test set
# and one explicit failure-case probe. The failure case is kept because it explains why
# answerability-aware evaluation is needed.

def print_rag_output(title, question):
    out = answerability_aware_rag_with_risk(question)

    print("=" * 90)
    print(title)
    print("=" * 90)

    print("QUESTION")
    print(out["question"])

    print("\nDECISION")
    print(out["decision"])

    print("\nANSWERABILITY PROBABILITY")
    print(round(out["prob_answerable"], 4))

    print("\nHALLUCINATION RISK PROXY")
    print(out.get("hallucination_risk_proxy", np.nan))

    print("\nANSWER")
    print(out["answer"])

    print("\nTOP RETRIEVED DOCUMENTS")
    for r in out["retrieved"]:
        print(f"- Rank {r['rank']} | {r['filename']} | score={r['hybrid_score']:.3f}")

    print("\nUNSUPPORTED CLAIMS FROM SUPPORT PROXY")
    for claim in out["support"].get("unsupported_claims", []):
        print("-", claim)

# Cleaner demo chosen from the already evaluated test results.
if "test_results" in globals() and len(test_results):
    clean_demo_pool = test_results[
        (test_results["true_answerable"] == True) &
        (test_results["decision"] == "answer") &
        (test_results["retrieval_hit"] == 1) &
        (
            test_results["unsupported_claim_rate"].isna() |
            (test_results["unsupported_claim_rate"] <= 0.5)
        )
    ].sort_values("prob_answerable", ascending=False)

    if len(clean_demo_pool):
        clean_question = clean_demo_pool.iloc[0]["question"]
    else:
        clean_question = df[df["answerable"]].iloc[0]["question"]
else:
    clean_question = df[df["answerable"]].iloc[0]["question"]

print_rag_output("Clean headline demo", clean_question)

# Keep this as a failure-case probe, not as the main demo.
# It helps demonstrate that related retrieval is not always sufficient evidence.
failure_question = "How do I configure SSL mutual authentication in IBM HTTP Server?"
print_rag_output("Failure-case probe: related context may still be insufficient", failure_question)

# %%  # Original notebook code cell 33
# Cell 26 — Performance metrics matrix (aligned to full test evaluation)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

# 1) Retrieval recall@K
# This is separate from retrieval_hit stored in test_results because it evaluates the retriever directly.
def retrieval_recall_at_k(split_df, top_k=TOP_K):
    hits = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Retrieval recall@{top_k}", leave=False):
        if not bool(row["answerable"]):
            continue

        gold = set()
        for ctx in row["contexts"]:
            if isinstance(ctx, dict) and ctx.get("filename"):
                gold.add(str(ctx["filename"]))
        if not gold:
            continue

        retrieved = retrieve(row["question"], top_k=top_k)
        retrieved_files = set(r["filename"] for r in retrieved)
        hits.append(int(len(gold.intersection(retrieved_files)) > 0))

    return float(np.mean(hits)) if hits else np.nan

val_recall = retrieval_recall_at_k(val_df, top_k=TOP_K)
test_recall = retrieval_recall_at_k(test_df, top_k=TOP_K)

# 2) Answerability classifier metrics, using the same calibrated probabilities used by the gate.
_, val_prob = calibrated_answerability_prob(X_val)
_, test_prob = calibrated_answerability_prob(X_test)
val_pred = (val_prob >= 0.50).astype(int)
test_pred = (test_prob >= 0.50).astype(int)

val_p, val_r, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)

# 3) End-to-end system metrics from full test_results
if "test_results" not in globals():
    test_results = evaluate_system(test_df, name="test", max_examples=MAX_EVAL_EXAMPLES)

sys_true = test_results["true_answerable"].astype(int).values
sys_pred = (test_results["decision"] == "answer").astype(int).values

unanswerable_df = test_results[test_results["true_answerable"] == False]
answerable_df = test_results[test_results["true_answerable"] == True]

unsafe_answer_rate = (unanswerable_df["decision"] == "answer").mean() if len(unanswerable_df) else np.nan
false_abstention_rate = (answerable_df["decision"] != "answer").mean() if len(answerable_df) else np.nan
retrieval_hit_rate = test_results.loc[test_results["retrieval_hit"].notna(), "retrieval_hit"].mean()

metrics_matrix = pd.DataFrame([
    {"component": f"retrieval@{TOP_K}", "split": "validation", "metric": "recall@k", "value": val_recall},
    {"component": f"retrieval@{TOP_K}", "split": "test",       "metric": "recall@k", "value": test_recall},

    {"component": "answerability_clf_calibrated", "split": "validation", "metric": "accuracy",  "value": accuracy_score(y_val, val_pred)},
    {"component": "answerability_clf_calibrated", "split": "validation", "metric": "auroc",     "value": roc_auc_score(y_val, val_prob)},
    {"component": "answerability_clf_calibrated", "split": "validation", "metric": "precision", "value": val_p},
    {"component": "answerability_clf_calibrated", "split": "validation", "metric": "recall",    "value": val_r},
    {"component": "answerability_clf_calibrated", "split": "validation", "metric": "f1",        "value": val_f1},

    {"component": "answerability_clf_calibrated", "split": "test", "metric": "accuracy",  "value": accuracy_score(y_test, test_pred)},
    {"component": "answerability_clf_calibrated", "split": "test", "metric": "auroc",     "value": roc_auc_score(y_test, test_prob)},
    {"component": "answerability_clf_calibrated", "split": "test", "metric": "precision", "value": test_p},
    {"component": "answerability_clf_calibrated", "split": "test", "metric": "recall",    "value": test_r},
    {"component": "answerability_clf_calibrated", "split": "test", "metric": "f1",        "value": test_f1},

    {"component": "end_to_end_system", "split": "test", "metric": "rows_evaluated",              "value": len(test_results)},
    {"component": "end_to_end_system", "split": "test", "metric": "decision_accuracy",           "value": accuracy_score(sys_true, sys_pred)},
    {"component": "end_to_end_system", "split": "test", "metric": "unsafe_answer_rate",          "value": unsafe_answer_rate},
    {"component": "end_to_end_system", "split": "test", "metric": "false_abstention_rate",       "value": false_abstention_rate},
    {"component": "end_to_end_system", "split": "test", "metric": "retrieval_hit_rate",          "value": retrieval_hit_rate},
    {"component": "end_to_end_system", "split": "test", "metric": "mean_rougeL",                 "value": test_results["rougeL"].mean()},
    {"component": "end_to_end_system", "split": "test", "metric": "mean_unsupported_claim_rate", "value": test_results["unsupported_claim_rate"].mean()},
]).sort_values(["component", "split", "metric"]).reset_index(drop=True)

display(metrics_matrix.style.format({"value": "{:.4f}"}))

# %%  # Original notebook code cell 34
# Cell 27 — Performance visualizations (aligned to same metrics/components)
# Note: this cell uses calibrated probabilities everywhere to match Cell 26 and the
# decision gate used at evaluation time. Earlier versions of this notebook used
# raw uncalibrated predictions here, which produced plots inconsistent with the
# metrics table. Fixed to use calibrated_answerability_prob throughout.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set(style="whitegrid")

# Use calibrated probabilities thresholded at 0.50 for the classifier-level plots,
# matching the metrics matrix computed in Cell 26.
_, val_prob_for_plot = calibrated_answerability_prob(X_val)
_, test_prob_for_plot = calibrated_answerability_prob(X_test)
val_pred_for_plot = (val_prob_for_plot >= 0.50).astype(int)
test_pred_for_plot = (test_prob_for_plot >= 0.50).astype(int)

# System decision labels (these already use the calibrated gate via answerability_aware_rag)
sys_true = test_results["true_answerable"].astype(int).values
sys_pred = (test_results["decision"] == "answer").astype(int).values


# ==============================================================================
# 1) Retrieval recall bars
# ==============================================================================
plt.figure(figsize=(8, 6))
ret_plot = metrics_matrix[
    (metrics_matrix["component"].str.startswith("retrieval")) &
    (metrics_matrix["metric"] == "recall@k")
].copy()

sns.barplot(data=ret_plot, x="split", y="value", palette="Blues_d")
plt.title(f"Retrieval Recall@{TOP_K}")
plt.ylim(0, 1)
plt.ylabel("Recall")
plt.xlabel("Split")

for i, v in enumerate(ret_plot["value"].values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()


# ==============================================================================
# 2) Answerability classifier confusion matrix (test, calibrated)
# ==============================================================================
plt.figure(figsize=(8, 6))
cm_clf = confusion_matrix(y_test, test_pred_for_plot)
sns.heatmap(
    cm_clf, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=["pred_unanswerable", "pred_answerable"],
    yticklabels=["true_unanswerable", "true_answerable"]
)
plt.title("Answerability Classifier Confusion Matrix (Test, calibrated @ 0.50)")
plt.xlabel("")
plt.ylabel("")

plt.tight_layout()
plt.show()


# ==============================================================================
# 3) ROC curve (test, calibrated)
# ==============================================================================
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, test_prob_for_plot)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#1f77b4", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.title("Answerability Classifier ROC (Test, calibrated)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# ==============================================================================
# 4) End-to-end decision confusion matrix
# ==============================================================================
plt.figure(figsize=(8, 6))
cm_sys = confusion_matrix(sys_true, sys_pred)
sns.heatmap(
    cm_sys, annot=True, fmt="d", cmap="Greens", cbar=False,
    xticklabels=["pred_should_not_answer", "pred_should_answer"],
    yticklabels=["true_should_not_answer", "true_should_answer"]
)
plt.title("End-to-End Decision Confusion Matrix")
plt.xlabel("")
plt.ylabel("")

plt.tight_layout()
plt.show()


# ==============================================================================
# 5) Compact bar chart for key system rates
# ==============================================================================
key = metrics_matrix[
    (metrics_matrix["component"] == "end_to_end_system") &
    (metrics_matrix["metric"].isin(["decision_accuracy", "unsafe_answer_rate", "false_abstention_rate", "mean_unsupported_claim_rate"]))
].copy()

plt.figure(figsize=(9, 4))
sns.barplot(data=key, x="metric", y="value", palette="Set2")
plt.ylim(0, 1)
plt.title("End-to-End Safety and Decision Quality")
plt.xticks(rotation=20)
plt.xlabel("")
plt.ylabel("Value")

for i, v in enumerate(key["value"].values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()

# %%  # Original notebook code cell 35
# Cell 28 — Stable BERTScore + appropriate visualization

import os
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"   # suppress HF auto-conversion thread noise

import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score as bertscore_score, BERTScorer

sns.set(style="whitegrid")

def _clean_text(x, max_chars=4000):
    x = re.sub(r"\s+", " ", str(x)).strip()
    return x[:max_chars]

# project-appropriate evaluation subset:
# answerable + actually answered by system
bdf = test_results.copy()
bdf = bdf[
    (bdf["true_answerable"] == True) &
    (bdf["decision"] == "answer") &
    bdf["gold_answer"].notna() &
    (bdf["gold_answer"].astype(str).str.strip() != "-")
].copy()

if len(bdf) == 0:
    print("No valid rows for BERTScore after filtering (answerable + decision=='answer').")
else:
    cands = [_clean_text(x) for x in bdf["generated_answer"].fillna("").tolist()]
    refs  = [_clean_text(x) for x in bdf["gold_answer"].tolist()]

    try:
        P, R, F1 = bertscore_score(
            cands, refs,
            model_type="microsoft/deberta-base-mnli",
            lang="en",
            rescale_with_baseline=False,     # keep scores in intuitive range
            use_fast_tokenizer=False,
            batch_size=8,
            verbose=True
        )
    except OverflowError:
        scorer = BERTScorer(
            model_type="microsoft/deberta-base-mnli",
            lang="en",
            rescale_with_baseline=False,
            use_fast_tokenizer=False,
            batch_size=8,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if hasattr(scorer, "_tokenizer") and getattr(scorer._tokenizer, "model_max_length", 0) > 100000:
            scorer._tokenizer.model_max_length = 512
        P, R, F1 = scorer.score(cands, refs)
        
    bdf["bertscore_precision"] = P.cpu().numpy()
    bdf["bertscore_recall"] = R.cpu().numpy()
    bdf["bertscore_f1"] = F1.cpu().numpy()

    display(bdf[["id", "decision", "bertscore_precision", "bertscore_recall", "bertscore_f1"]].head())

    print("Mean BERTScore Precision:", round(float(bdf["bertscore_precision"].mean()), 4))
    print("Mean BERTScore Recall    :", round(float(bdf["bertscore_recall"].mean()), 4))
    print("Mean BERTScore F1        :", round(float(bdf["bertscore_f1"].mean()), 4))

    # ==============================================================================
    # Visualization 1: BERTScore F1 Distribution
    # ==============================================================================
    plt.figure(figsize=(8, 5))
    sns.histplot(bdf["bertscore_f1"], bins=15, kde=True, color="#2a9d8f")
    plt.title("BERTScore F1 Distribution (Answered+Answerable)")
    plt.xlabel("BERTScore F1")
    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # Visualization 2: BERTScore F1 vs ROUGE-L
    # ==============================================================================
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=bdf, x="rougeL", y="bertscore_f1", color="#457b9d")
    corr = bdf[["rougeL", "bertscore_f1"]].dropna().corr().iloc[0, 1] if bdf["rougeL"].notna().any() else np.nan
    plt.title(f"BERTScore F1 vs ROUGE-L (corr={corr:.3f})" if not np.isnan(corr) else "BERTScore F1 vs ROUGE-L")
    plt.xlabel("ROUGE-L")
    plt.ylabel("BERTScore F1")
    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # Visualization 3: BERTScore Component Spread
    # ==============================================================================
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=bdf[["bertscore_precision", "bertscore_recall", "bertscore_f1"]])
    plt.title("BERTScore Component Spread")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# %%  # Original notebook code cell 36
# Cell 28b — Gate Confidence vs. Output Quality Correlation
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

if "bdf" in globals() and "bertscore_f1" in bdf.columns:
    plt.figure(figsize=(9, 6))
    
    # Use a regression plot to show the trendline
    ax = sns.regplot(
        data=bdf, 
        x="prob_answerable", 
        y="bertscore_f1", 
        scatter_kws={"alpha": 0.7, "color": "#457b9d", "s": 50}, 
        line_kws={"color": "#e76f51", "linewidth": 2}
    )
    
    plt.title("System Confidence vs. Actual Generation Quality")
    plt.xlabel("Answerability Gate Confidence (Probability)")
    plt.ylabel("BERTScore F1")
    plt.xlim(0.5, 1.05) # Assuming answered questions mostly score > 0.5
    plt.tight_layout()
    plt.show()
    
    # Calculate and print Pearson correlation
    corr = bdf["prob_answerable"].corr(bdf["bertscore_f1"])
    print(f"Correlation between Gate Confidence and Output Quality: {corr:.3f}")
else:
    print("Run the BERTScore cell (Cell 28) first so 'bdf' is available.")

# %%  # Original notebook code cell 37
# Cell 29 — LLM Judge Evaluation + Visualization (robust + fixed)

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

sns.set(style="whitegrid")

JUDGE_N = min(100, len(test_results))
judge_df = test_results.sample(n=JUDGE_N, random_state=SEED).reset_index(drop=True).copy()

GRADE_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

DIMENSIONS = {
    "decision_appropriateness": (
        "Was the action/decision appropriate? "
        "If question is unanswerable, abstain/request_more_evidence should score high. "
        "If answerable, answering should score high."
    ),
    "factuality": "How factually correct is the system answer vs reference/context?",
    "completeness": "How completely does the system answer address the question?",
    "overall": "Overall quality considering decision, factuality, and completeness."
}

def _clip(x, n=700):
    return re.sub(r"\s+", " ", str(x)).strip()[:n]

@torch.inference_mode()
def _generate(text, max_new_tokens=20):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    device = next(gen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        temperature=0.0
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # 🔥 IMPORTANT: Remove prompt echo
    return decoded.replace(text, "").strip()

def _extract_score(raw_text):
    t = str(raw_text).strip().upper()

    # ✅ XML tag
    m = re.search(r"<SCORE>\s*([ABCDE])\s*</SCORE>", t)
    if m:
        return GRADE_TO_SCORE[m.group(1)]

    # ✅ "Score: B"
    m = re.search(r"SCORE[:\s\-]*([ABCDE])", t)
    if m:
        return GRADE_TO_SCORE[m.group(1)]

    # ✅ standalone grade anywhere
    m = re.search(r"\b([ABCDE])\b", t)
    if m:
        return GRADE_TO_SCORE[m.group(1)]

    # ✅ numeric anywhere
    m = re.search(r"\b([1-5])\b", t)
    if m:
        return int(m.group(1))

    return np.nan

def _judge_dimension(question, gold_answer, system_answer, true_answerable, dim_name, dim_desc):
    ref = gold_answer if true_answerable else "N/A (unanswerable question)"

    prompt = f"""
You are a strict evaluator.

Criterion: {dim_name}
Definition: {dim_desc}

Question answerable: {true_answerable}

System answer:
{_clip(system_answer, 400)}

Reference:
{_clip(ref, 400)}

Grade using ONLY one letter: A, B, C, D, or E

Output:
<score>X</score>
""".strip()

    txt = _generate(prompt, max_new_tokens=20)
    score = _extract_score(txt)

    # 🔁 Retry if failed
    if np.isnan(score):
        retry_prompt = f"""
Grade with ONE letter only.

A=best, E=worst.

Answer:
{_clip(system_answer, 300)}

Return ONLY:
<score>X</score>
""".strip()

        txt2 = _generate(retry_prompt, max_new_tokens=10)
        score = _extract_score(txt2)

        # If parsing still fails, return NaN so the row is treated as missing
        # rather than artificially pulling the mean toward the neutral score.
        # The parse rate is reported below as a diagnostic.
        if np.isnan(score):
            return np.nan, txt2

        return score, txt2

    # ✅ Always return something
    return score, txt

def llm_judge_once(row):
    out = {}
    for dim, desc in DIMENSIONS.items():
        score, raw = _judge_dimension(
            row["question"], row["gold_answer"], row["generated_answer"], bool(row["true_answerable"]),
            dim, desc
        )
        out[dim] = score
        out[f"raw_{dim}"] = raw
    return out

# Run judge
rows = []
for _, r in tqdm(judge_df.iterrows(), total=len(judge_df), desc="LLM judge"):
    rows.append(llm_judge_once(r))

score_df = pd.concat([judge_df, pd.DataFrame(rows)], axis=1)
cols = ["decision_appropriateness", "factuality", "completeness", "overall"]
for c in cols:
    score_df[c] = pd.to_numeric(score_df[c], errors="coerce")

# Parse diagnostics
score_df["all_parsed"] = score_df[cols].notna().all(axis=1)
parse_rate = score_df["all_parsed"].mean()
print(f"Parsed complete rows: {score_df['all_parsed'].sum()}/{len(score_df)} ({parse_rate:.1%})")
print("\nPer-dimension parse rate:")
print(score_df[cols].notna().mean().round(3))
print("\nMean scores (on available parsed values):")
print(score_df[cols].mean().round(3))

# ==============================================================================
# Improved Visualization 1: Mean Scores with Confidence Intervals
# ==============================================================================
plot_data = score_df[cols].melt(var_name="Dimension", value_name="Score").dropna()

if len(plot_data):
    plt.figure(figsize=(9, 5))
    # Using a horizontal bar plot with error bars (95% CI)
    sns.barplot(data=plot_data, y="Dimension", x="Score", palette="viridis", capsize=0.1)
    plt.xlim(1, 5.2) # Give a little padding for the error bars
    plt.title("LLM Judge Mean Scores (with 95% Confidence Intervals)")
    plt.xlabel("Mean Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
else:
    print("\n[Notice] No parsable judge scores available for the bar plot.")

# ==============================================================================
# Improved Visualization 2: Overall Score Distribution by Decision (Box + Strip)
# ==============================================================================
box_df = score_df.dropna(subset=["overall"]).copy()

if len(box_df):
    plt.figure(figsize=(9, 5))
    # Base boxplot (white inside to let the dots stand out)
    sns.boxplot(data=box_df, x="decision", y="overall", color="white", showfliers=False)
    # Overlay individual data points
    sns.stripplot(data=box_df, x="decision", y="overall", palette="magma", size=7, jitter=True, alpha=0.7, hue="decision", legend=False)
    
    plt.ylim(0.5, 5.5)
    plt.title("Judge Overall Score by Decision (with individual evaluations)")
    plt.xlabel("System Decision")
    plt.ylabel("Overall Score (1-5)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
else:
    print("\n[Notice] No valid overall scores available to plot the boxplot.")

# ==============================================================================
# New Visualization 3: Dimension Correlation Heatmap
# ==============================================================================
# Only run if we have more than 1 row of complete data
if len(score_df.dropna(subset=cols)) > 1:
    corr_df = score_df[cols].dropna().corr()
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Between Judge Dimensions")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("\n[Notice] Not enough valid data to generate a correlation heatmap.")

# ==============================================================================
# Self-evaluation caveat
# ==============================================================================
# IMPORTANT: this LLM judge uses the same model (FLAN-T5-Large) that generated
# the system answers. Literature on LLM-as-judge consistently shows that models
# score their own outputs more favourably than a third-party model would. The
# scores above should therefore be treated as a weak internal signal, not as
# headline evidence of answer quality. Replacing the judge with a different
# (and ideally larger) model — for example Llama-3-8B-Instruct or a hosted
# frontier model — is left for future work and noted in the proposal under
# scope and limitations.
print("\n[Caveat] LLM judge uses the same model as the generator. Treat scores as")
print("weak internal signal, not as independent evaluation. See note in cell.")

# %%  # Original notebook code cell 38
# Cell 30 — Uncertainty Calibration (with isotonic post-hoc calibration) + Visualization
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

val_prob = answerability_clf.predict_proba(X_val)[:, 1]
test_prob_raw = answerability_clf.predict_proba(X_test)[:, 1]
y_true = y_test.astype(int)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(val_prob, y_val)
test_prob_cal = iso.transform(test_prob_raw)

def calibration_table(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        gap = abs(acc - conf)
        ece += (m.sum() / len(y)) * gap
        rows.append({"bin": b, "n": m.sum(), "confidence": conf, "accuracy": acc, "gap": gap})
    return pd.DataFrame(rows), ece

raw_tab, ece_raw = calibration_table(y_true, test_prob_raw, n_bins=10)
cal_tab, ece_cal = calibration_table(y_true, test_prob_cal, n_bins=10)

print(f"Raw Brier: {brier_score_loss(y_true, test_prob_raw):.4f} | Raw ECE: {ece_raw:.4f}")
print(f"Calibrated Brier: {brier_score_loss(y_true, test_prob_cal):.4f} | Calibrated ECE: {ece_cal:.4f}")

# ==============================================================================
# Visualization 1: Reliability Diagram
# ==============================================================================
plt.figure(figsize=(7, 5))
plt.plot([0, 1], [0, 1], "k--", label="Perfect")
plt.plot(raw_tab["confidence"], raw_tab["accuracy"], marker="o", label="Raw")
plt.plot(cal_tab["confidence"], cal_tab["accuracy"], marker="o", label="Isotonic-calibrated")
plt.title("Reliability Diagram")
plt.xlabel("Mean predicted probability")
plt.ylabel("Empirical accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
# Visualization 2: Bin Occupancy
# ==============================================================================
counts_df = pd.DataFrame({
    "raw_bin_count": raw_tab.set_index("bin")["n"],
    "cal_bin_count": cal_tab.set_index("bin")["n"]
}).fillna(0)

# Pandas .plot() will use the currently active figure if we set one up
plt.figure(figsize=(7, 5))
counts_df.plot(kind="bar", color=["#e76f51", "#2a9d8f"], ax=plt.gca())
plt.title("Bin Occupancy (Raw vs Calibrated)")
plt.xlabel("Bin index")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %%  # Original notebook code cell 39
# Cell 31 — Ablation Study + Visualization (full test split)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Use the full test split for dissertation-level reporting.
ablation_sample = test_df.reset_index(drop=True).copy()
print("Ablation rows:", len(ablation_sample))

def run_policy(question, ans_thr=ANSWER_THRESHOLD, req_thr=REQUEST_MORE_EVIDENCE_THRESHOLD, force_answer=False, binary_gate=False):
    retrieved = retrieve(question, top_k=TOP_K)
    p, _ = predict_answerability(question)

    if force_answer:
        decision = "answer"
    elif binary_gate:
        decision = "answer" if p >= ans_thr else "abstain"
    else:
        decision = "answer" if p >= ans_thr else ("request_more_evidence" if p >= req_thr else "abstain")

    if decision == "answer":
        answer = generate_answer(question, retrieved)
        if is_refusal_answer(answer) or len(str(answer).split()) < 4:
            fallback = extractive_context_answer(question, retrieved)
            if fallback:
                answer = fallback
            elif not force_answer:
                decision = "request_more_evidence"
                answer = "The retrieved documentation appears related but not sufficient. Please provide more specific or stronger evidence before answering."
        support = claim_support_analysis(answer, retrieved) if decision == "answer" else {"unsupported_rate": 0.0, "num_claims": 0}
    elif decision == "request_more_evidence":
        answer = "The retrieved documentation appears related but not sufficient. Please provide more specific or stronger evidence before answering."
        support = {"unsupported_rate": 0.0, "num_claims": 0}
    else:
        answer = "Unable to answer based on the provided documentation."
        support = {"unsupported_rate": 0.0, "num_claims": 0}

    return {"decision": decision, "answer": answer, "prob": p, "support": support}

def evaluate_variant(name, runner, df_eval):
    rows = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"Ablation: {name}"):
        out = runner(row["question"])
        answered = out["decision"] == "answer"
        true_ans = bool(row["answerable"])
        r = rouge_l(out["answer"], row["answer"]) if (true_ans and answered) else np.nan
        rows.append({
            "true_answerable": true_ans,
            "decision": out["decision"],
            "rougeL": r,
            "unsupported": out["support"].get("unsupported_rate", np.nan),
        })
    res = pd.DataFrame(rows)
    unans = res[res["true_answerable"] == False]
    ans = res[res["true_answerable"] == True]
    return {
        "variant": name,
        "rows": len(res),
        "decision_accuracy": ((res["decision"] == "answer").astype(int).eq(res["true_answerable"].astype(int))).mean(),
        "answer_coverage": (res["decision"] == "answer").mean(),
        "unsafe_answer_rate": (unans["decision"] == "answer").mean() if len(unans) else np.nan,
        "false_abstention_rate": (ans["decision"] != "answer").mean() if len(ans) else np.nan,
        "mean_rougeL": res["rougeL"].mean(),
        "mean_unsupported_claim_rate": res["unsupported"].mean(),
    }

variants = {
    f"Tuned full policy ({ANSWER_THRESHOLD:.2f}/{REQUEST_MORE_EVIDENCE_THRESHOLD:.2f})": lambda q: run_policy(q, ANSWER_THRESHOLD, REQUEST_MORE_EVIDENCE_THRESHOLD),
    "Conservative gate (0.70/0.50)": lambda q: run_policy(q, 0.70, 0.50),
    "Aggressive gate (0.50/0.30)": lambda q: run_policy(q, 0.50, 0.30),
    "Binary gate (answer/abstain)": lambda q: run_policy(q, ANSWER_THRESHOLD, REQUEST_MORE_EVIDENCE_THRESHOLD, binary_gate=True),
    "No gate (always answer)": lambda q: run_policy(q, force_answer=True),
}

ablation_rows = [evaluate_variant(name, fn, ablation_sample) for name, fn in variants.items()]
ablation_df = pd.DataFrame(ablation_rows)
display(ablation_df.round(4))

# ---------------------------------------------------------
# Visualization (Split by Polarity in Separate Frames)
# ---------------------------------------------------------
plot_df = ablation_df.melt(
    id_vars="variant",
    value_vars=["decision_accuracy", "answer_coverage", "unsafe_answer_rate", "false_abstention_rate", "mean_unsupported_claim_rate"],
    var_name="metric", value_name="value"
)

# Define metric categories
success_metrics = ["decision_accuracy", "answer_coverage"]
risk_metrics = ["unsafe_answer_rate", "false_abstention_rate", "mean_unsupported_claim_rate"]

# 1) Success Metrics Frame
plt.figure(figsize=(8, 5))
sns.barplot(
    data=plot_df[plot_df["metric"].isin(success_metrics)],
    x="metric", y="value", hue="variant", palette="viridis"
)
plt.title("Success and Coverage Metrics")
plt.ylim(0, 1.05)
plt.ylabel("Score / Rate")
plt.xlabel("")
plt.xticks(rotation=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy Variant")
plt.tight_layout()
plt.show()

# 2) Risk & Error Metrics Frame
plt.figure(figsize=(10, 5))
sns.barplot(
    data=plot_df[plot_df["metric"].isin(risk_metrics)],
    x="metric", y="value", hue="variant", palette="magma"
)
plt.title("Risk & Error Metrics (Lower is Better)")
plt.ylim(0, 1.05)
plt.ylabel("Score / Rate")
plt.xlabel("")
plt.xticks(rotation=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Policy Variant")
plt.tight_layout()
plt.show()

# %%  # Original notebook code cell 40
# Cell 32 — Bootstrap Confidence Intervals for Headline Metrics

# The TechQA-RAG-Eval test split is small (roughly 137 rows after a 70/15/15
# stratified split), and the minority answerability class is typically only
# 20-40 examples in test. A single point estimate of unsafe_answer_rate or
# false_abstention_rate on that sample size can move noticeably just by
# reshuffling. Bootstrap resampling gives a confidence interval around each
# headline metric so the reader can see the spread rather than just the mean.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(SEED)

def bootstrap_ci(values, statistic_fn, n_boot=N_BOOTSTRAP, ci=95):
    """Generic bootstrap CI for a statistic computed on a 1D array."""
    values = np.asarray(values)
    values = values[~pd.isna(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot_stats = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot_stats[i] = statistic_fn(values[idx])
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(statistic_fn(values)), float(lo), float(hi)

def bootstrap_paired_ci(df_in, statistic_fn, n_boot=N_BOOTSTRAP, ci=95):
    """Bootstrap CI when the statistic needs multiple columns from a dataframe."""
    df_in = df_in.reset_index(drop=True)
    n = len(df_in)
    if n == 0:
        return np.nan, np.nan, np.nan
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sample = df_in.iloc[idx]
        boot_stats[i] = statistic_fn(sample)
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(statistic_fn(df_in)), float(lo), float(hi)

# Statistics for the end-to-end test results
def stat_decision_accuracy(df_in):
    true = df_in["true_answerable"].astype(int).values
    pred = (df_in["decision"] == "answer").astype(int).values
    return accuracy_score(true, pred)

def stat_unsafe_rate(df_in):
    unans = df_in[df_in["true_answerable"] == False]
    if len(unans) == 0:
        return np.nan
    return (unans["decision"] == "answer").mean()

def stat_false_abstention_rate(df_in):
    ans = df_in[df_in["true_answerable"] == True]
    if len(ans) == 0:
        return np.nan
    return (ans["decision"] != "answer").mean()

def stat_retrieval_hit_rate(df_in):
    s = df_in.loc[df_in["retrieval_hit"].notna(), "retrieval_hit"]
    if len(s) == 0:
        return np.nan
    return s.mean()

def stat_mean_rougeL(df_in):
    s = df_in["rougeL"].dropna()
    if len(s) == 0:
        return np.nan
    return s.mean()

def stat_mean_unsupported(df_in):
    s = df_in["unsupported_claim_rate"].dropna()
    if len(s) == 0:
        return np.nan
    return s.mean()

print(f"Running {N_BOOTSTRAP} bootstrap resamples on {len(test_results)} test rows...")

system_ci_rows = []
for name, fn in [
    ("decision_accuracy", stat_decision_accuracy),
    ("unsafe_answer_rate", stat_unsafe_rate),
    ("false_abstention_rate", stat_false_abstention_rate),
    ("retrieval_hit_rate", stat_retrieval_hit_rate),
    ("mean_rougeL", stat_mean_rougeL),
    ("mean_unsupported_claim_rate", stat_mean_unsupported),
]:
    point, lo, hi = bootstrap_paired_ci(test_results, fn)
    system_ci_rows.append({
        "metric": name,
        "point_estimate": point,
        "ci_low_95": lo,
        "ci_high_95": hi,
        "ci_width": hi - lo if not (np.isnan(hi) or np.isnan(lo)) else np.nan,
    })

system_ci_df = pd.DataFrame(system_ci_rows)
print("\nEnd-to-end system metrics with 95% bootstrap CIs:")
display(system_ci_df.round(4))

# Bootstrap CIs for the answerability classifier on test
_, test_prob_calibrated_for_ci = calibrated_answerability_prob(X_test)
test_pred_for_ci = (test_prob_calibrated_for_ci >= 0.50).astype(int)

clf_eval_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": test_pred_for_ci,
    "y_prob": test_prob_calibrated_for_ci,
})

def stat_clf_accuracy(df_in):
    return accuracy_score(df_in["y_true"], df_in["y_pred"])

def stat_clf_auroc(df_in):
    if df_in["y_true"].nunique() < 2:
        return np.nan
    return roc_auc_score(df_in["y_true"], df_in["y_prob"])

def stat_clf_f1(df_in):
    _, _, f1, _ = precision_recall_fscore_support(
        df_in["y_true"], df_in["y_pred"], average="binary", zero_division=0
    )
    return f1

clf_ci_rows = []
for name, fn in [
    ("classifier_accuracy", stat_clf_accuracy),
    ("classifier_auroc", stat_clf_auroc),
    ("classifier_f1", stat_clf_f1),
]:
    point, lo, hi = bootstrap_paired_ci(clf_eval_df, fn)
    clf_ci_rows.append({
        "metric": name,
        "point_estimate": point,
        "ci_low_95": lo,
        "ci_high_95": hi,
        "ci_width": hi - lo if not (np.isnan(hi) or np.isnan(lo)) else np.nan,
    })

clf_ci_df = pd.DataFrame(clf_ci_rows)
print("\nAnswerability classifier metrics with 95% bootstrap CIs:")
display(clf_ci_df.round(4))

# Combined plot
import matplotlib.pyplot as plt
combined_ci = pd.concat([system_ci_df, clf_ci_df], ignore_index=True)

plt.figure(figsize=(10, 6))
y_pos = np.arange(len(combined_ci))
plt.errorbar(
    combined_ci["point_estimate"],
    y_pos,
    xerr=[
        combined_ci["point_estimate"] - combined_ci["ci_low_95"],
        combined_ci["ci_high_95"] - combined_ci["point_estimate"]
    ],
    fmt="o", color="#1f77b4", ecolor="#888888", capsize=4, markersize=8
)
plt.yticks(y_pos, combined_ci["metric"])
plt.xlabel("Value (with 95% bootstrap CI)")
plt.title("Headline Metrics with Bootstrap Confidence Intervals")
plt.axvline(0.5, color="lightgray", linestyle=":", linewidth=1)
plt.xlim(0, 1.05)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nNote: CI widths are sensitive to test split size (n =", len(test_results),
      "rows, with imbalanced minority class). Wide intervals reflect the small")
print("evaluation sample rather than weakness of the underlying method.")

