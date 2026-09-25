"""
rq3_core.py - RQ3: certainty from query-time signals.

RQ3. Can a composite certainty score built only from signals available at query time,
with no access to gold annotations, place each answer into a defined certainty band, and how
well does that banding separate correct answers from incorrect ones?

Two parts, kept apart because they rest on different evidence.

A. Retrieval-side certainty, at full scale and out of sample. For each of the 910 labelled
   questions, signals are read from the centre-point retrieval alone: how similar the top
   segment is, how far it stands above the rest, whether dense and lexical retrieval agree,
   how concentrated the context is. The target is whether the answer's evidence reached the
   context - RQ2's reachable partition, with unanswerable questions counted as having none.
   RQ2 found no correct answer outside that partition, so a score that predicts it defines the
   "insufficient evidence" band. The score is fitted on one split and tested on the other, in
   both directions, with every setting fixed in advance.

B. Generation-side certainty on the forty reviewed answers: the three-band rule over
   abstention and citations, recomputed from the answer text. In-sample, as before.

Signal computation needs rq1_core (passed in as a module); the analysis needs only numpy and
pandas, and the logistic model is fitted by Newton's method here so that results do not drift
with library versions.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "1.0"

SEED = 42
N_BOOT = 2000
S3_THRESHOLD = 0.80
L2 = 1.0                                   # fixed in advance, on standardised signals
BAND_CUTS = (0.35, 0.60, 0.85)             # the four bands of the design
BANDS = ["insufficient", "low", "probable", "certain"]

SIGNALS = ["dense_top1", "dense_gap12", "dense_top5_mean", "dense_z_top1", "bm25_z_top1",
           "agree_top5", "agree_top1_doc", "ctx_docs", "top1_doc_share",
           "log_query_tokens", "log_candidates"]
RERANK_SIGNALS = ["rerank_top1", "rerank_gap12", "rerank_std"]
# A signal is admissible only if a deployed system could compute it from the question and its
# own ranking. The candidate pool size fails that test: it is the length of the benchmark's
# per-question candidate list, an artefact of how TechQA was built, with no counterpart in a
# deployment that searches the whole corpus. It is still computed and still reported in the
# audit table, but it is kept out of the model.
EXCLUDED_SIGNALS = {
    "log_candidates": "size of the benchmark's candidate list; no deployment counterpart"}
MODEL_SIGNALS = [s for s in SIGNALS if s not in EXCLUDED_SIGNALS]
SIGNAL_NAMES = {
    "dense_top1": "top-segment similarity", "dense_gap12": "gap to the second segment",
    "dense_top5_mean": "mean similarity of the context", "dense_z_top1": "top-segment prominence",
    "bm25_z_top1": "lexical prominence", "agree_top5": "dense-lexical context agreement",
    "agree_top1_doc": "dense-lexical top-document agreement", "ctx_docs": "documents in context",
    "top1_doc_share": "top-document share of context", "log_query_tokens": "question length",
    "log_candidates": "candidate pool size", "rerank_top1": "top reranker score",
    "rerank_gap12": "reranker gap", "rerank_std": "reranker dispersion"}


# =================================================================================
# A1. Signals (Colab, needs the cached index)
# =================================================================================

def compute_signals(C, ds, idx, reranker=None, k=5, progress=True):
    """One row per labelled question: deployment signals, then gold-derived labels.

    Labels are computed after the signals from the same ranking and are prefixed `label_`;
    nothing prefixed `label_` may be used as a signal.
    """
    rows, pairs, owners = [], [], []
    iterator = ds.questions
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(ds.questions, desc="signals")
        except Exception:
            pass
    for q in iterator:
        qid = q["QUESTION_ID"]
        query = C.make_query(q)
        cand = idx.candidate_rows(q)
        row = {"question_id": qid, "split": q["SPLIT"], "answerable": C.is_answerable(q)}
        if len(cand) < 2:
            rows.append({**row, **{s: np.nan for s in SIGNALS},
                         "label_stage": "S1" if row["answerable"] else "U",
                         "label_evidence": 0.0})
            continue
        dense = np.asarray(idx.dense_scores(q, cand), dtype=np.float64)
        order = np.lexsort((cand, -dense))
        ranked, dsort = cand[order], dense[order]
        top = ranked[:k]
        lex = np.asarray(idx.bm25.scores(query, cand), dtype=np.float64)
        ltop = cand[np.lexsort((cand, -lex))][:k]
        docs = idx.doc_id[top]
        row.update({
            "dense_top1": dsort[0],
            "dense_gap12": dsort[0] - dsort[1],
            "dense_top5_mean": dsort[:k].mean(),
            "dense_z_top1": (dsort[0] - dense.mean()) / (dense.std() + 1e-9),
            "bm25_z_top1": (lex.max() - lex.mean()) / (lex.std() + 1e-9),
            "agree_top5": len(set(top) & set(ltop)) / len(set(top) | set(ltop)),
            "agree_top1_doc": float(idx.doc_id[top[0]] == idx.doc_id[ltop[0]]),
            "ctx_docs": float(len(set(docs))),
            "top1_doc_share": float(np.mean(docs == docs[0])),
            "log_query_tokens": math.log1p(len(query.split())),
            "log_candidates": math.log1p(len(cand)),
        })
        # ---- labels: gold annotations, evaluation only --------------------------------
        if row["answerable"]:
            m = C.evidence_metrics(idx, q, ranked)
            in20, in5 = m["gold_document_in_top_20"] > 0, m["gold_document_in_top_5"] > 0
            cov = m["gold_span_coverage"]
            # RQ3 needs only whether the evidence arrived, and the reachable set is the same
            # under either stage ordering. Which stage blocked a question is RQ2's business
            # and needs the segmentation ceiling, so it is not decided here: B is "blocked".
            stage = "R" if (in20 and in5 and cov >= S3_THRESHOLD) else "B"
        else:
            stage = "U"
        row["label_stage"] = stage
        row["label_evidence"] = float(stage == "R")
        rows.append(row)
        if reranker is not None:
            for r in top:
                pairs.append((query, idx.chunk_text(int(r))))
                owners.append(qid)
    df = pd.DataFrame(rows)
    if reranker is not None and pairs:
        scores = np.asarray(reranker.predict(pairs, batch_size=32,
                                             show_progress_bar=progress), dtype=np.float64)
        by_q = pd.DataFrame({"question_id": owners, "s": scores})
        agg = by_q.groupby("question_id")["s"].agg(
            rerank_top1="max",
            rerank_gap12=lambda s: (np.sort(s.values)[::-1][0] - np.sort(s.values)[::-1][1])
            if len(s) > 1 else 0.0,
            rerank_std="std").reset_index()
        df = df.merge(agg, on="question_id", how="left")
    return df


# =================================================================================
# A2. Statistics
# =================================================================================

def auroc(score, y):
    """Mann-Whitney AUROC with ties counted as one half."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(bool)
    keep = ~np.isnan(s)
    s, y = s[keep], y[keep]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_auroc(score, y, n_boot=N_BOOT, seed=SEED):
    s = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(bool)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        a = auroc(s[i], y[i])
        if not math.isnan(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return auroc(s, y), float(lo), float(hi)


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins=10):
    """Expected calibration error over equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "observed": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


# =================================================================================
# A3. The composite score
# =================================================================================

def _prepare(train, test, features):
    Xtr = train[features].to_numpy(dtype=float)
    Xte = test[features].to_numpy(dtype=float)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, {"median": med, "mean": mu, "std": sd}


def logistic_fit(X, y, l2=L2, iters=100, tol=1e-10):
    """L2-penalised logistic regression by Newton's method; the intercept is not penalised."""
    X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ w, -35, 35)
        p = 1 / (1 + np.exp(-z))
        grad = X.T @ (p - y) + pen * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def logistic_predict(w, X):
    z = np.clip(np.column_stack([np.ones(len(X)), X]) @ w, -35, 35)
    return 1 / (1 + np.exp(-z))


def band_of(p, cuts=BAND_CUTS):
    return np.array(BANDS)[np.digitize(np.asarray(p, float), cuts)]


def band_table(p, frame):
    b = band_of(p)
    rows = []
    for band in BANDS:
        m = b == band
        g = frame[m]
        rows.append({"band": band, "n": int(m.sum()), "share": float(m.mean()),
                     "mean_score": float(np.mean(p[m])) if m.any() else np.nan,
                     "evidence_rate": float(g.label_evidence.mean()) if m.any() else np.nan,
                     "unanswerable": int((g.label_stage == "U").sum()),
                     # "B", or the older per-stage letters in signal files written before
                     # the stage order was corrected: anything that is neither R nor U
                     "blocked": int((~g.label_stage.isin(["R", "U"])).sum()),
                     "reachable": int((g.label_stage == "R").sum())})
    return pd.DataFrame(rows)


def selective(p, y, cut=BAND_CUTS[0]):
    """Withhold every question scored below `cut`."""
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    held = p < cut
    return {"cut": cut, "withheld_share": float(held.mean()),
            "caught": float(held[~y].mean()) if (~y).any() else np.nan,
            "lost": float(held[y].mean()) if y.any() else np.nan,
            "evidence_rate_answered": float(y[~held].mean()) if (~held).any() else np.nan,
            "evidence_rate_all": float(y.mean())}


def evaluate(signals, features, fit_split, test_split, n_boot=N_BOOT):
    train = signals[signals.split == fit_split]
    test = signals[signals.split == test_split]
    Xtr, Xte, stats = _prepare(train, test, features)
    w = logistic_fit(Xtr, train.label_evidence.to_numpy())
    p = logistic_predict(w, Xte)
    y = test.label_evidence.to_numpy()
    auc, lo, hi = boot_auroc(p, y, n_boot)
    top1, t_lo, t_hi = boot_auroc(test["dense_top1"].to_numpy(), y, n_boot)
    base = float(train.label_evidence.mean())
    b_model, b_base = brier(p, y), brier(np.full(len(y), base), y)
    ans = test.answerable.to_numpy().astype(bool)
    # Base-rate shift: move every score by the log-odds difference between the two splits'
    # evidence rates. It changes no ranking and uses a single aggregate figure from the test
    # split - the rate a deployment would have to estimate - and no per-question label.
    rate_t = float(y.mean())
    shift = (math.log(rate_t / (1 - rate_t)) - math.log(base / (1 - base))
             if 0 < rate_t < 1 and 0 < base < 1 else 0.0)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    p_shift = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + shift)))
    return {
        "fit": fit_split, "test": test_split, "n_fit": int(len(train)), "n_test": int(len(test)),
        "features": list(features),
        "coefficients": {f: float(c) for f, c in zip(["intercept"] + list(features), w)},
        "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
        "auroc_top1_only": top1, "auroc_top1_lo": t_lo, "auroc_top1_hi": t_hi,
        "auroc_answerable_only": auroc(p[ans], y[ans]),
        "brier": b_model, "brier_base_rate": b_base,
        "brier_skill": float(1 - b_model / b_base) if b_base > 0 else np.nan,
        "ece": ece(p, y), "base_rate_fit": base, "base_rate_test": float(y.mean()),
        "bands": band_table(p, test).to_dict(orient="records"),
        "selective": selective(p, y),
        "reliability": reliability(p, y).to_dict(orient="records"),
        "mean_score": float(p.mean()),
        "unanswerable_fit": float((~train.answerable.astype(bool)).mean()),
        "unanswerable_test": float((~test.answerable.astype(bool)).mean()),
        "base_rate_shift": {
            "log_odds_shift": float(shift),
            "ece": ece(p_shift, y), "brier": brier(p_shift, y),
            "auroc": auroc(p_shift, y),
            "bands": band_table(p_shift, test).to_dict(orient="records"),
            "selective": selective(p_shift, y),
            "reliability": reliability(p_shift, y).to_dict(orient="records"),
        },
        "_p": p, "_y": y, "_p_shift": p_shift,
    }


def univariate(signals, features, split="dev", n_boot=N_BOOT):
    d = signals[signals.split == split]
    y = d.label_evidence.to_numpy()
    rows = []
    for f in features:
        a, lo, hi = boot_auroc(d[f].to_numpy(), y, n_boot)
        # orientation-free strength, so a signal that runs the other way is not hidden
        rows.append({"signal": f, "name": SIGNAL_NAMES.get(f, f), "auroc": a,
                     "lo": lo, "hi": hi, "strength": max(a, 1 - a) if not math.isnan(a) else a,
                     "direction": "higher means evidence present" if a >= 0.5
                     else "higher means evidence absent"})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


# =================================================================================
# B. Generation-side bands on the reviewed answers
# =================================================================================

CHUNK_ID = re.compile(r"([A-Za-z0-9]+)::chunk_(\d+)")
TRUNCATED = re.compile(r"\[([A-Za-z]+[0-9A-Za-z]*)\s*$")


def cited_segments(answer):
    """Distinct cited segments: every complete identifier, of any Technote series (swg..., nas...),
    plus a citation cut off by the generation limit at the very end of the answer. The truncated
    one counts as a citation that cannot be matched to any document, since the model did cite."""
    a = str(answer)
    ids = [f"{d}::chunk_{n}" for d, n in CHUNK_ID.findall(a)]
    m = TRUNCATED.search(a)
    if m and len(m.group(1)) > 3:
        ids.append(f"{m.group(1)}::truncated")
    return list(dict.fromkeys(ids))


def generation_bands(reviewed, is_abstention):
    """The three-band rule of the certainty study, recomputed from the answer text.

    low     abstained, cited nothing, or cited three or more segments
    high    cited one or two segments, including one from the top-ranked document
    medium  otherwise
    """
    rows = []
    for _, r in reviewed.iterrows():
        ctx = [c.strip() for c in str(r.top_5_chunk_ids).split("|") if c.strip()]
        top_doc = ctx[0].split("::")[0] if ctx else None
        cited = cited_segments(r.predicted_answer)
        n = len(cited)
        cites_top = any(c.split("::")[0] == top_doc for c in cited)
        abstained = bool(is_abstention(str(r.predicted_answer)))
        if abstained or n == 0 or n >= 3:
            band = "low"
        elif cites_top:
            band = "high"
        else:
            band = "medium"
        label = str(r.manual_answer_correct).strip().lower()
        rows.append({"question_id": r.question_id, "band": band, "n_cited": n,
                     "cites_top_document": cites_top, "abstained": abstained,
                     "label": label,
                     "credit": {"correct": 1.0, "partial": 0.5}.get(label, 0.0)})
    return pd.DataFrame(rows)


def generation_band_table(gb, stages=None):
    g = gb if stages is None else gb[gb.question_id.isin(stages)]
    rows = []
    for band in ("high", "medium", "low"):
        s = g[g.band == band]
        rows.append({"band": band, "n": int(len(s)),
                     "correct": int((s.label == "correct").sum()),
                     "partial": int((s.label == "partial").sum()),
                     "incorrect": int((s.label == "incorrect").sum()),
                     "credit": float(s.credit.mean()) if len(s) else np.nan})
    return pd.DataFrame(rows)


# =================================================================================
# Figures
# =================================================================================

# The validated categorical palette, in its documented order, shared with rq2_core so a colour
# means the same thing across the dissertation's figures.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, RULE, MUTED = "#ffffff", "#3f3d39", "#8d8a83"
FIG_STYLE = {
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 8, "axes.labelsize": 9,
    "axes.labelcolor": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#c9c6bf", "axes.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 8, "legend.handlelength": 1.2, "legend.columnspacing": 1.4,
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update(FIG_STYLE)
    return plt


def plot_reliability(results, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, key, colour in zip(axes, ("train_to_dev", "dev_to_train"), (PALETTE[0], PALETTE[1])):
        r = results[key]
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=MUTED, lw=1.0,
                label="perfect calibration")
        curves = [(r["reliability"], "o", colour, "fitted score", 1.0)]
        shifted = r.get("base_rate_shift")
        if shifted:
            curves.append((shifted["reliability"], "s", RULE,
                           "base rate set to the test split", 0.75))
        biggest = max(pd.DataFrame(c[0]).n.max() for c in curves)
        for table, marker, col, label, alpha in curves:
            rel = pd.DataFrame(table).dropna()
            sizes = 18 + 240 * rel.n / biggest
            face = col if marker == "o" else "none"
            ax.scatter(rel.mean_predicted, rel.observed, s=sizes, marker=marker,
                       facecolor=face, edgecolor=col if marker == "s" else SURFACE, lw=1.2,
                       alpha=alpha, zorder=3)
            ax.plot(rel.mean_predicted, rel.observed, color=col, lw=2.0, alpha=alpha,
                    ls="-" if marker == "o" else "--", label=label)
        for c in BAND_CUTS:
            ax.axvline(c, color="#e6e4df", lw=0.9, ls="-", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("certainty score")
        ax.set_ylabel("share of questions whose evidence reached the context")
        ece_txt = (f"ECE {r['ece']:.3f} -> {shifted['ece']:.3f} after the base-rate shift"
                   if shifted else f"ECE {r['ece']:.3f}")
        ax.set_title(f"fitted on {r['fit']}, tested on {r['test']}\n"
                     f"AUROC {r['auroc']:.3f}, {ece_txt}", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Calibration of the retrieval-side certainty score (marker size = questions)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_bands(results, path):
    plt = _plt()
    b = pd.DataFrame(results["train_to_dev"]["bands"])
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(b))
    parts = [("reachable", PALETTE[2], "evidence reached the context"),
             ("blocked", PALETTE[1], "answerable, evidence lost"),
             ("unanswerable", MUTED, "unanswerable")]
    bottom = np.zeros(len(b))
    for col, colour, label in parts:
        v = b[col].to_numpy(dtype=float)
        ax.bar(x, v, bottom=bottom, color=colour, edgecolor=SURFACE, lw=1.2, width=0.7,
               label=label)
        bottom += v
    for xi, (n, rate) in enumerate(zip(b.n, b.evidence_rate)):
        if n:
            ax.text(xi, bottom[xi] + 2, f"{100 * rate:.0f}% with evidence", ha="center",
                    fontsize=8, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{band}\n(n={n})" for band, n in zip(b.band, b.n)])
    ax.set_ylabel("development questions")
    ax.set_title("What each certainty band contains (fitted on training, development split)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =================================================================================
# Orchestration
# =================================================================================

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict(orient="records"))
    return o if (o is None or isinstance(o, str)) else str(o)


def cross_fitted_scores(signals, features):
    """Every question's retrieval-side score from the model fitted on the other split, before
    and after the base-rate shift, so that no question is scored by a model that saw it."""
    out = []
    for fit, test in (("train", "dev"), ("dev", "train")):
        tr = signals[signals.split == fit]
        te = signals[signals.split == test]
        if not len(tr) or not len(te):
            continue
        Xtr, Xte, _ = _prepare(tr, te, features)
        p = logistic_predict(logistic_fit(Xtr, tr.label_evidence.to_numpy()), Xte)
        base, rate = float(tr.label_evidence.mean()), float(te.label_evidence.mean())
        shift = (math.log(rate / (1 - rate)) - math.log(base / (1 - base))
                 if 0 < rate < 1 and 0 < base < 1 else 0.0)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        ps = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + shift)))
        out.append(pd.DataFrame({"question_id": te.question_id.values, "score": p,
                                 "score_shifted": ps}))
    return pd.concat(out, ignore_index=True)


def answer_bands(answers, is_abstention, answer_body):
    """The three-band citation rule applied to generated answers.

    `answers` has question_id, config_id, answer and segment_ids (the context, in rank order,
    joined by '|'); the top-ranked document is the document of the first segment.
    """
    rows = []
    for r in answers.itertuples():
        raw = str(r.answer)
        body = answer_body(raw)
        segs = [x for x in str(r.segment_ids).split("|") if x]
        top_doc = segs[0].split("::")[0] if segs else None
        cited = cited_segments(raw)
        n = len(cited)
        cites_top = any(c.split("::")[0] == top_doc for c in cited)
        abstained = bool(is_abstention(body) or is_abstention(raw))
        if abstained or n == 0 or n >= 3:
            band = "low"
        elif cites_top:
            band = "high"
        else:
            band = "medium"
        rows.append({"question_id": r.question_id, "config_id": r.config_id, "band": band,
                     "n_cited": n, "cites_top_document": cites_top, "abstained": abstained})
    return pd.DataFrame(rows)


GEN_BANDS = ["low", "medium", "high"]


def citation_test(answers, scored, is_abstention, answer_body, threshold, signals=None,
                  features=None, exclude=(), centre="sliding_window|dense|0", n_boot=N_BOOT,
                  seed=SEED):
    """Out-of-sample test of the citation rule on generated answers.

    An answer to an answerable question counts as correct when its token F1 against the gold
    span reaches `threshold`; an answer to an unanswerable question is a false answer unless it
    abstains. Questions in `exclude` (those the rule was derived from) are left out.
    """
    b = answer_bands(answers, is_abstention, answer_body)
    sc = scored.drop_duplicates(["question_id", "config_id"])[
        ["question_id", "config_id", "answerable", "token_f1"]]
    m = b.merge(sc, on=["question_id", "config_id"])
    m["answerable"] = m.answerable.astype(str).str.lower().isin(["true", "1", "1.0"])
    m = m[~m.question_id.isin(set(exclude))].copy()
    m["correct"] = m.answerable & (pd.to_numeric(m.token_f1, errors="coerce") >= threshold)
    ans = m[m.answerable]

    def table(d):
        rows = []
        for band in GEN_BANDS:
            g = d[d.band == band]
            rows.append({"band": band, "n": int(len(g)),
                         "correct": float(g.correct.mean()) if len(g) else float("nan")})
        return rows

    def high_low(d):
        """High-band minus low-band correct share, with an interval that resamples whole
        questions, since one question contributes an answer under every configuration."""
        hi_m, lo_m = d.band == "high", d.band == "low"
        if not hi_m.any() or not lo_m.any():
            return None
        qs = d.question_id.unique()
        by_q = {q: g for q, g in d.groupby("question_id")}
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_boot):
            sample = pd.concat([by_q[q] for q in rng.choice(qs, len(qs))])
            h = sample[sample.band == "high"].correct
            l_ = sample[sample.band == "low"].correct
            if len(h) and len(l_):
                diffs.append(h.mean() - l_.mean())
        hi_v, lo_v = float(d[hi_m].correct.mean()), float(d[lo_m].correct.mean())
        return {"high": hi_v, "low": lo_v, "diff": hi_v - lo_v,
                "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}

    rank = {"low": 0, "medium": 1, "high": 2}
    c = ans[ans.config_id == centre]
    res = {
        "threshold": float(threshold), "excluded_questions": len(set(exclude)),
        "questions": int(m.question_id.nunique()),
        "answerable_questions": int(ans.question_id.nunique()),
        "answers": int(len(m)),
        "centre": {"bands": table(c), "high_vs_low": high_low(c),
                   "auroc": auroc(c.band.map(rank).to_numpy(float), c.correct.to_numpy(float))},
        "pooled": {"bands": table(ans), "high_vs_low": high_low(ans),
                   "auroc": auroc(ans.band.map(rank).to_numpy(float),
                                  ans.correct.to_numpy(float))},
        "per_configuration": [
            {"config_id": k, **{f"{r['band']}_n": r["n"] for r in table(g)},
             **{f"{r['band']}_correct": r["correct"] for r in table(g)}}
            for k, g in ans.groupby("config_id")],
        "unanswerable_low_band": float((m[~m.answerable].band == "low").mean())
        if (~m.answerable).any() else float("nan"),
        "answerable_low_band": float((ans.band == "low").mean()),
    }

    if signals is not None and features:
        s = cross_fitted_scores(signals, features)
        cs = c.merge(s, on="question_id")
        allc = m[m.config_id == centre].merge(s, on="question_id")
        res["retrieval_score"] = {
            "auroc_correct_answerable": auroc(cs.score.to_numpy(), cs.correct.to_numpy(float)),
            "auroc_correct_with_unanswerable": auroc(allc.score.to_numpy(),
                                                     allc.correct.to_numpy(float)),
        }
        allc["retrieval_band"] = [band_of(x) for x in allc.score_shifted]
        allc["gen_band"] = allc.band
        combo = []
        for rb in BANDS:
            for gb in GEN_BANDS:
                g = allc[(allc.retrieval_band == rb) & (allc.gen_band == gb)]
                if len(g):
                    combo.append({"retrieval_band": rb, "citation_band": gb, "n": int(len(g)),
                                  "correct": float(g.correct.mean()),
                                  "unanswerable": float((~g.answerable).mean())})
        res["joint_bands_centre"] = combo
        flagged = allc[(allc.retrieval_band == "insufficient") | (allc.gen_band == "low")]
        kept = allc.drop(flagged.index)
        res["joint_rule_centre"] = {
            "withheld_share": float(len(flagged) / len(allc)) if len(allc) else float("nan"),
            "correct_if_kept": float(kept.correct.mean()) if len(kept) else float("nan"),
            "correct_all": float(allc.correct.mean()),
            "unanswerable_caught": float((~flagged.answerable).sum() / max(1, (~allc.answerable).sum())),
            "correct_lost": float(flagged.correct.sum() / max(1, allc.correct.sum())),
        }
    return res, m


def add_citation_test(results_dir, answers_csv, scored_csv, reviewed_csv, is_abstention,
                      answer_body, threshold, n_boot=1000):
    """Run citation_test on notebook D's answers and store it in rq3_results.json."""
    out = Path(results_dir)
    res = json.loads((out / "rq3_results.json").read_text(encoding="utf-8"))
    signals = pd.read_csv(out / "rq3_signals.csv")
    reviewed = pd.read_csv(reviewed_csv)
    ct, rows = citation_test(pd.read_csv(answers_csv), pd.read_csv(scored_csv), is_abstention,
                             answer_body, threshold, signals=signals, features=res["features"],
                             exclude=set(reviewed.question_id), n_boot=n_boot)
    res["citation_test"] = ct
    rows.to_csv(out / "rq3_citation_test_answers.csv", index=False)
    (out / "rq3_results.json").write_text(json.dumps(jsonable(res), indent=2, allow_nan=False),
                                          encoding="utf-8")
    return ct


def run_analysis(signals, out_dir, reviewed=None, is_abstention=None, n_boot=N_BOOT):
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    present = [f for f in SIGNALS + RERANK_SIGNALS
               if f in signals.columns and signals[f].notna().any()]
    features = [f for f in present if f not in EXCLUDED_SIGNALS]
    ans = signals[signals.answerable.astype(bool)]
    res = {
        "n_questions": int(len(signals)),
        "n_by_split": signals.split.value_counts().to_dict(),
        "features": features,
        "signals_computed": present,
        "excluded_signals": {f: EXCLUDED_SIGNALS[f] for f in present if f in EXCLUDED_SIGNALS},
        "reranker_used": any(f in features for f in RERANK_SIGNALS),
        "evidence_rate": float(signals.label_evidence.mean()),
        "stage_shares_answerable": ans.label_stage.value_counts(normalize=True).to_dict(),
        "unanswerable": int((signals.label_stage == "U").sum()),
        "train_to_dev": evaluate(signals, features, "train", "dev", n_boot),
        "dev_to_train": evaluate(signals, features, "dev", "train", n_boot),
        # the audit table keeps every computed signal, marked by whether the model may use it
        "univariate": univariate(signals, present, "dev", n_boot).assign(
            used=lambda t: ~t.signal.isin(EXCLUDED_SIGNALS)),
    }
    plot_reliability(res, out / "figures" / "fig_rq3_reliability.png")
    plot_bands(res, out / "figures" / "fig_rq3_bands.png")
    pd.DataFrame(res["train_to_dev"]["bands"]).to_csv(out / "rq3_bands_dev.csv", index=False)
    res["univariate"].to_csv(out / "rq3_univariate_dev.csv", index=False)
    if reviewed is not None and is_abstention is not None:
        gb = generation_bands(reviewed, is_abstention)
        s = reviewed.copy()
        b = s.gold_document_in_top_20.astype(str).str.lower().isin(["true", "1", "1.0"])
        c = s.gold_document_in_top_5.astype(str).str.lower().isin(["true", "1", "1.0"])
        reach = set(s.question_id[b & c & (s.gold_span_coverage.astype(float) >= S3_THRESHOLD)])
        res["generation"] = {
            "all": generation_band_table(gb),
            "reachable": generation_band_table(gb, reach),
            "n_reachable": len(reach),
            "high_vs_low_credit": [float(gb[gb.band == "high"].credit.mean()),
                                   float(gb[gb.band == "low"].credit.mean())],
        }
        gb.to_csv(out / "rq3_generation_bands.csv", index=False)
    signals.to_csv(out / "rq3_signals.csv", index=False)
    (out / "rq3_results.json").write_text(json.dumps(jsonable(res), indent=2, allow_nan=False),
                                          encoding="utf-8")
    return res


# =================================================================================
# C. The answer to RQ3, read from the results
# =================================================================================

def answer(res, width=78):
    """Print RQ3's answer from the computed results. Every number is read, none is written
    here, so the statement cannot drift from the run that produced it."""
    t, d = res["train_to_dev"], res["dev_to_train"]
    bands = {b["band"]: b for b in t["bands"]}
    ordered = [bands[b]["evidence_rate"] for b in BANDS if b in bands]
    monotone = all(x <= y + 1e-9 for x, y in zip(ordered, ordered[1:]))
    ct = res.get("citation_test", {})
    lines = ["=" * width,
             "THE ANSWER TO RQ3",
             "Can a certainty score built only from signals available at query time, with no",
             "gold annotation, place answers in bands that separate right from wrong?",
             "=" * width,
             "",
             f"Signals used            {len(res['features'])} "
             f"({', '.join(sorted(res.get('excluded_signals', {}))) or 'none'} excluded: "
             f"{'; '.join(res.get('excluded_signals', {}).values()) or 'nothing excluded'})",
             f"Ranking, out of sample  AUROC {t['auroc']:.3f} "
             f"[{t['auroc_lo']:.3f}, {t['auroc_hi']:.3f}] fitted on {t['fit']}, "
             f"and {d['auroc']:.3f} the other way round",
             f"                        against {t['auroc_top1_only']:.3f} for the top segment's "
             f"similarity alone",
             f"Probabilities           ECE {t['ece']:.3f}, and "
             f"{t['base_rate_shift']['ece']:.3f} once the score's base rate is set to the "
             f"split it serves",
             f"Bands                   share with evidence "
             f"{' -> '.join(f'{100 * x:.0f}%' for x in ordered)} "
             f"({'rises with every band' if monotone else 'NOT monotone'})"]
    if ct:
        c = ct["centre"]["bands"]
        by = {b["band"]: b for b in c}
        lines += [f"Citation rule, new questions  high {100 * by['high']['correct']:.0f}% pass "
                  f"against low {100 * by['low']['correct']:.0f}%, on "
                  f"{ct['answerable_questions']} questions it was not derived from "
                  f"(band AUROC {ct['centre']['auroc']:.2f})"]
    lines += ["",
              "Answered IN PART.",
              "  Yes: the ranking holds on questions the model never saw, so the lowest band",
              "  is a usable 'do not trust this answer' signal, and the thresholds transfer",
              "  once set to the base rate of the questions being served.",
              "  No: the probabilities are not calibrated out of the box, and the citation",
              "  rule separates the upper bands only weakly, so the bands support a decision",
              "  to abstain rather than a guarantee of correctness.",
              "",
              "  Note: no system was deployed in this study. The claim is that these signals",
              "  need no gold annotation and exist as soon as an answer is produced.",
              "=" * width]
    print("\n".join(lines))
    return {"auroc_out_of_sample": t["auroc"], "monotone_bands": monotone,
            "verdict": "in part"}
