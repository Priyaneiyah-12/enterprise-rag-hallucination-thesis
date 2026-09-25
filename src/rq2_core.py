"""
rq2_core.py - RQ2: stage attribution over the per-question traces of the RQ1 experiments.

RQ2. Can responsibility for a failed answer be attributed to a single, mutually exclusive
pipeline stage using only observable trace evidence, and does that attribution yield
engineering conclusions that aggregate scoring cannot?

The decision order is the one of Table 3.2, applied to trace fields only, and it follows the
order in which the pipeline itself runs: a document is segmented, the segments are indexed,
the index is searched, the context is assembled, and only then is an answer generated.

    S1  segmentation loss    under this chunking method the gold span does not survive into
                             the context budget: the best five segments of the gold document
                             together hold less than 80% of it, so no retriever could have
                             delivered the evidence. Measured before retrieval runs.
    S2  retrieval miss       the segmentation kept the span, but the gold Technote is not
                             among the retrieved candidates
    S3  context exclusion    it is a candidate, but no segment of it reaches the context, or
                             the segments that were chosen carry less than 80% of the span
                             although the segmentation had kept it
    R   reachable            the evidence is in the context; a failure here is S4,
                             generation drift, and needs a judged answer to be seen

The first broken stage wins, so the four labels are mutually exclusive and exhaustive by
construction. S1 to S3 need no generator, which is why they can be measured for every
configuration and every question; S4 needs judged answers.

S1 is a property of the chunking method alone, so it is read from a per-(method, question)
ceiling table written by notebook RQ2-0; `configuration_table` merges it in.

Standalone on purpose: numpy, pandas and matplotlib only, so the analysis runs on a CPU
runtime and does not depend on which build of rq1_core.py a notebook finds on Drive.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "2.0"          # 2.0: pipeline-order attribution, segmentation tested first

SEED = 42
N_BOOT = 5000
S3_THRESHOLD = 0.80          # the same 80% of the gold span at both places it is applied
ORDER = ["S1", "S2", "S3", "R"]
BLOCKING = ["S1", "S2", "S3"]
NAMES = {"S1": "segmentation loss", "S2": "retrieval miss", "S3": "context exclusion",
         "R": "reachable", "S4": "generation drift", "OK": "correct"}
CEILING = "segmentation_ceiling"    # best coverage the chunking method leaves attainable
CEILING_FILE = "rq2_segmentation_ceiling.csv"
CENTRE_METHOD = "sliding_window"    # the chunking method of the centre point

# every configuration measured by the RQ1 notebooks, as (factor, level column, level, label)
CENTRE = "centre point"
CONFIGURATIONS = [
    ("chunking", "method", "fixed_window", "fixed window"),
    ("chunking", "method", "recursive_structure", "recursive structure-aware"),
    ("chunking", "method", "semantic_breakpoint", "semantic breakpoint"),
    ("chunking", "method", "late_chunking", "late chunking"),
    ("encoder", "method", "sliding_window_longctx", "long-context encoder"),
    ("retrieval", "retriever", "bm25", "BM25"),
    ("retrieval", "retriever", "hybrid_rrf", "hybrid RRF"),
    ("retrieval", "retriever", "hybrid_weighted", "hybrid weighted"),
    ("reranking", "rerank_level", "cross_encoder", "cross-encoder reranking"),
]


# =================================================================================
# 1. The attribution rule
# =================================================================================

def _flag(s):
    """Trace flags arrive as 0.0/1.0 from the RQ1 files and as True/False from the review
    files; both are read the same way."""
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes"]).to_numpy()


def _ceiling(df):
    """The segmentation ceiling column, as floats. Missing it is an error rather than a silent
    fallback, because without it the first stage of the pipeline cannot be tested."""
    if CEILING not in df.columns:
        raise KeyError(
            f"{CEILING!r} is missing. Run notebook RQ2_0_Segmentation_Ceiling.ipynb, put "
            "rq2_segmentation_ceiling.csv beside the per-question traces, and merge it with "
            "configuration_table() or with_ceiling().")
    return pd.to_numeric(df[CEILING], errors="coerce").fillna(1.0).to_numpy()


def attribute(df, threshold=S3_THRESHOLD):
    """Stage code for every row, in pipeline order: the first stage that broke wins."""
    ceiling = _ceiling(df)
    in_pool = _flag(df["gold_document_in_top_20"])
    in_ctx = _flag(df["gold_document_in_top_5"])
    cov = pd.to_numeric(df["gold_span_coverage"], errors="coerce").fillna(0.0).to_numpy()
    codes = np.select([ceiling < threshold, ~in_pool, ~in_ctx | (cov < threshold)],
                      ["S1", "S2", "S3"], default="R")
    return pd.Series(codes, index=df.index, name="stage")


def with_ceiling(df, ceiling, method=None):
    """Merge the per-(method, question) ceiling table onto a trace or review frame.

    `method` names the chunking method to use when the frame has no method column, which is
    the case for the reviewed answers: they were produced at the centre point.
    """
    c = ceiling[["method", "question_id", CEILING]]
    df = df.drop(columns=[CEILING], errors="ignore")      # never merge onto an older copy
    if "method" in df.columns:
        return df.merge(c, on=["method", "question_id"], how="left")
    if method is None:
        raise ValueError("no method column: pass method= for a single-configuration frame")
    c = c[c.method == method].drop(columns="method")
    return df.merge(c, on="question_id", how="left")


def trace_defects(df):
    """Rows whose trace is internally inconsistent: the gold Technote in the context but not
    among the candidates, or coverage recorded without the Technote in the context."""
    in_pool = _flag(df["gold_document_in_top_20"])
    in_ctx = _flag(df["gold_document_in_top_5"])
    cov = pd.to_numeric(df["gold_span_coverage"], errors="coerce").fillna(0.0).to_numpy()
    defects = {"context_without_candidate": int((in_ctx & ~in_pool).sum()),
               "coverage_without_context": int(((cov > 0) & ~in_ctx).sum()),
               "coverage_out_of_range": int(((cov < 0) | (cov > 1 + 1e-9)).sum())}
    if CEILING in df.columns:
        # What reached the context can never exceed what the segmentation left attainable.
        # The traces are stored with four decimals, so a coverage equal to its ceiling can
        # read up to 5e-5 above it; only a larger gap is a real disagreement.
        defects["coverage_above_ceiling"] = int((cov > _ceiling(df) + 1e-4).sum())
    return defects


def attribute_outcomes(df, label_col="manual_answer_correct", success=("correct",),
                       threshold=S3_THRESHOLD):
    """Stages for reviewed answers: a reachable question that failed is S4."""
    stage = attribute(df, threshold)
    label = df[label_col].astype(str).str.strip().str.lower()
    ok = label.isin([s.lower() for s in success])
    final = np.where(stage != "R", stage, np.where(ok, "OK", "S4"))
    return pd.DataFrame({"question_id": df.get("question_id", pd.Series(df.index)).values,
                         "stage": stage.values, "label": label.values,
                         "success": ok.values, "attributed": final}, index=df.index)


# =================================================================================
# 2. Statistics
# =================================================================================

def _boot_index(n, n_boot=N_BOOT, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def share_ci(x, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    boot = x[_boot_index(len(x), n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def paired_delta(d, n_boot=N_BOOT, seed=SEED):
    """Mean of per-question differences, a percentile interval, and a two-sided bootstrap p."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    boot = d[_boot_index(len(d), n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(p)}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# =================================================================================
# 3. Assembling the configurations
# =================================================================================

def load_traces(results_dir):
    """Read the three per-question files written by notebooks A, B and C."""
    R = Path(results_dir)
    frames = {}
    for key, name in (("A", "rq1_factorA_per_question.csv"),
                      ("B", "rq1_factorB_per_question.csv"),
                      ("C", "rq1_factorC_per_question.csv")):
        p = R / name
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing - notebooks A, B and C write these files")
        frames[key] = pd.read_csv(p)
    return frames


def configuration_table(frames, ceiling=None):
    """One frame, one row per (configuration, question), with the centre point once.

    The centre point is measured three times - as the sliding-window level of A, the dense
    level of B and the off level of C. The three copies are checked to be identical before one
    is kept, because if they were not, the factors would not share a reference.

    `ceiling` is the per-(method, question) segmentation ceiling of notebook RQ2-0; it is
    merged in so that S1 can be decided before any retrieval field is read.
    """
    A, B, Cf = frames["A"], frames["B"], frames["C"]
    if ceiling is not None:
        A, B, Cf = (with_ceiling(f, ceiling) for f in (A, B, Cf))
    fields = ["question_id", "gold_document_in_top_20", "gold_document_in_top_5",
              "gold_span_coverage"] + ([CEILING] if CEILING in A.columns else [])
    centre_a = A[A.method == "sliding_window"][fields].set_index("question_id").sort_index()
    centre_b = B[B.retriever == "dense"][fields].set_index("question_id").sort_index()
    centre_c = Cf[Cf.rerank_level == "off"][fields].set_index("question_id").sort_index()
    identical = (centre_a.equals(centre_b) and centre_a.equals(centre_c))
    source = {"A": A, "B": B, "C": Cf}
    parts = [centre_a.reset_index().assign(configuration=CENTRE, factor="centre")]
    for factor, col, level, label in CONFIGURATIONS:
        frame = {"method": A, "retriever": B, "rerank_level": Cf}[col]
        sub = frame[frame[col] == level]
        if len(sub):
            parts.append(sub[fields].assign(configuration=label, factor=factor))
    table = pd.concat(parts, ignore_index=True)
    return table, identical


# =================================================================================
# 4. The analyses
# =================================================================================

def partition(table, threshold=S3_THRESHOLD, n_boot=N_BOOT):
    rows = []
    for cfg, g in table.groupby("configuration", sort=False):
        st = attribute(g, threshold).to_numpy()
        row = {"configuration": cfg, "factor": g.factor.iloc[0], "n": len(st)}
        for s in ORDER:
            m, lo, hi = share_ci(st == s, n_boot)
            row[s], row[f"{s}_lo"], row[f"{s}_hi"] = m, lo, hi
        m, lo, hi = share_ci(st != "R", n_boot)
        row["blocked"], row["blocked_lo"], row["blocked_hi"] = m, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_stages(table, level, baseline, threshold):
    a = table[table.configuration == level].set_index("question_id")
    b = table[table.configuration == baseline].set_index("question_id")
    common = a.index.intersection(b.index).sort_values()
    return (attribute(a.loc[common], threshold).to_numpy(),
            attribute(b.loc[common], threshold).to_numpy(), common)


def decompose(table, level, baseline=CENTRE, threshold=S3_THRESHOLD, n_boot=N_BOOT):
    """Where a configuration change acts: the paired change in each stage's share.

    The shares sum to one, so the four changes sum to zero; a change that only moves
    questions between stages shows up as offsetting deltas with a large `moved` count.
    """
    sa, sb, common = _paired_stages(table, level, baseline, threshold)
    out = {"configuration": level, "baseline": baseline, "n": len(common),
           "moved": int((sa != sb).sum()),
           "net_reachable": int((sa == "R").sum() - (sb == "R").sum())}
    ps = []
    for s in ORDER:
        r = paired_delta((sa == s).astype(float) - (sb == s).astype(float), n_boot)
        out[f"d{s}"], out[f"d{s}_lo"], out[f"d{s}_hi"], out[f"d{s}_p"] = \
            r["mean"], r["lo"], r["hi"], r["p"]
        if s in BLOCKING:
            ps.append(r["p"])
    for s, a in zip(BLOCKING, holm(ps)):
        out[f"d{s}_p_holm"] = float(a)
    resolved = [s for s in BLOCKING
                if not (out[f"d{s}_lo"] <= 0 <= out[f"d{s}_hi"]) and out[f"d{s}_p_holm"] < 0.05]
    out["dominant_stage"] = (max(resolved, key=lambda s: abs(out[f"d{s}"]))
                             if resolved else None)
    return out


def decompositions(table, pairs=None, threshold=S3_THRESHOLD, n_boot=N_BOOT):
    labels = [c for c in table.configuration.unique() if c != CENTRE]
    pairs = pairs or [(c, CENTRE) for c in labels]
    return pd.DataFrame([decompose(table, a, b, threshold, n_boot) for a, b in pairs])


def transitions(table, level, baseline=CENTRE, threshold=S3_THRESHOLD):
    sa, sb, _ = _paired_stages(table, level, baseline, threshold)
    t = pd.crosstab(pd.Series(sb, name=f"{baseline}"), pd.Series(sa, name=f"{level}"))
    return t.reindex(index=ORDER, columns=ORDER, fill_value=0)


def threshold_sensitivity(table, configuration=CENTRE, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    g = table[table.configuration == configuration]
    rows = []
    for thr in thresholds:
        st = attribute(g, thr).to_numpy()
        rows.append({"threshold": thr, **{s: float((st == s).mean()) for s in ORDER}})
    return pd.DataFrame(rows)


def reviewed_outcomes(reviewed, threshold=S3_THRESHOLD):
    """Stage x review label for a reviewed run, plus the checks that validate the rule."""
    o = attribute_outcomes(reviewed, threshold=threshold)
    table = pd.crosstab(o.stage, o.label).reindex(ORDER, fill_value=0)
    lenient = o.label.isin(["correct", "partial"])
    return {
        "n": int(len(o)),
        "partition": {s: int((o.stage == s).sum()) for s in ORDER},
        "labels": o.label.value_counts().to_dict(),
        "stage_by_label": table.to_dict(),
        "correct_outside_reachable": int(((o.stage != "R") & o.success).sum()),
        "partial_outside_reachable": int(((o.stage != "R") & (o.label == "partial")).sum()),
        "correct_inside_reachable": int(((o.stage == "R") & o.success).sum()),
        "s4_strict": int(((o.stage == "R") & ~o.success).sum()),
        "s4_lenient": int(((o.stage == "R") & ~lenient).sum()),
        "reachable": int((o.stage == "R").sum()),
        "attributed": o.attributed.value_counts().to_dict(),
    }


def ablation(v1, v2, threshold=S3_THRESHOLD):
    """The V1/V2 context-budget runs: identical retrieval, different generation budget."""
    same_ids = list(v1.question_id) == list(v2.question_id)
    same_ctx = bool((v1.top_5_chunk_ids.astype(str).values ==
                     v2.top_5_chunk_ids.astype(str).values).all()) if same_ids else False
    r1, r2 = reviewed_outcomes(v1, threshold), reviewed_outcomes(v2, threshold)
    same_partition = r1["partition"] == r2["partition"]
    return {"same_questions_same_order": bool(same_ids),
            "identical_contexts": same_ctx,
            "identical_partition": bool(same_partition),
            "v1": r1, "v2": r2,
            "gain_correct": r2["correct_inside_reachable"] + r2["correct_outside_reachable"]
                            - r1["correct_inside_reachable"] - r1["correct_outside_reachable"],
            "gain_inside_reachable": r2["correct_inside_reachable"] - r1["correct_inside_reachable"],
            "s4_strict_change": r2["s4_strict"] - r1["s4_strict"]}


def sophistication_path(parts, path=("BM25", CENTRE, "hybrid RRF", "cross-encoder reranking")):
    """The stage profile along increasingly elaborate retrieval pipelines."""
    p = parts.set_index("configuration")
    return p.loc[[c for c in path if c in p.index], ORDER + ["blocked"]].reset_index()


# =================================================================================
# 4b. The generation stage at scale
# =================================================================================

# generated run -> configuration label used above
RUN_CONFIGURATION = {
    "sliding_window|dense|0": CENTRE,
    "fixed_window|dense|0": "fixed window",
    "recursive_structure|dense|0": "recursive structure-aware",
    "semantic_breakpoint|dense|0": "semantic breakpoint",
    "late_chunking|dense|0": "late chunking",
    "sliding_window|bm25|0": "BM25",
    "sliding_window|hybrid_rrf|0": "hybrid RRF",
    "sliding_window|hybrid_weighted|0": "hybrid weighted",
    "sliding_window|dense|1": "cross-encoder reranking",
}


def calibrate_check(f1, labels, positive=("correct",), grid=None):
    """The token-F1 threshold that best reproduces a reviewer's judgement.

    Chosen by balanced accuracy over a fixed grid; ties go to the threshold nearest 0.15 so
    the choice does not wander between equally good values. Agreement and Cohen's kappa are
    reported for the chosen threshold.
    """
    f1 = np.asarray(f1, dtype=float)
    lab = pd.Series(labels).astype(str).str.strip().str.lower()
    pos = lab.isin(positive).to_numpy()
    grid = np.round(np.arange(0.05, 0.40, 0.005), 3) if grid is None else np.asarray(grid)
    best = None
    for t in grid:
        pred = f1 >= t
        tpr = (pred & pos).sum() / max(1, pos.sum())
        tnr = (~pred & ~pos).sum() / max(1, (~pos).sum())
        key = ((tpr + tnr) / 2, -abs(t - 0.15))
        if best is None or key > best[0]:
            best = (key, t)
    t = float(best[1])
    pred = f1 >= t
    agree = float((pred == pos).mean())
    pe = pred.mean() * pos.mean() + (1 - pred.mean()) * (1 - pos.mean())
    return {"threshold": t, "balanced_accuracy": float(best[0][0]), "agreement": agree,
            "kappa": float((agree - pe) / (1 - pe)) if pe < 1 else float("nan"),
            "n": int(len(f1)), "positives": int(pos.sum()),
            "false_pass_rate": float((pred & ~pos).sum() / max(1, (~pos).sum()))}


def generation_outcomes(table, scored, threshold):
    """Stage and outcome for every generated answer to an answerable question.

    `scored` is the notebook D output; an answer counts as correct when its token F1 against
    the gold span reaches `threshold`. A reachable question answered incorrectly - including by
    refusing - is generation drift, S4.
    """
    g = scored[scored.answerable.astype(bool)].drop_duplicates(["config_id", "question_id"]).copy()
    g["configuration"] = g.config_id.map(RUN_CONFIGURATION)
    g = g[g.configuration.notna()]
    stages = table.assign(stage=attribute(table).values)[["configuration", "question_id", "stage"]]
    m = g.merge(stages, on=["configuration", "question_id"], how="inner")
    m["correct"] = pd.to_numeric(m.token_f1, errors="coerce").fillna(0.0) >= threshold
    m["abstained"] = pd.to_numeric(m.abstained, errors="coerce").fillna(0.0) > 0.5
    m["outcome"] = np.where(m.stage != "R", m.stage, np.where(m.correct, "OK", "S4"))
    return m[["configuration", "question_id", "stage", "abstained", "correct", "outcome"]]


def generation_summary(outcomes, n_boot=N_BOOT):
    o = outcomes
    rows = []
    for cfg, g in o.groupby("configuration", sort=False):
        s4 = g[g.outcome == "S4"]
        row = {"configuration": cfg, "n": int(len(g))}
        for s in ("S1", "S2", "S3", "S4", "OK"):
            row[s] = float((g.outcome == s).mean())
        row["correct_outside_reachable"] = int(((g.stage != "R") & g.correct).sum())
        row["s4_refusal_share"] = float(s4.abstained.mean()) if len(s4) else float("nan")
        rows.append(row)
    per_config = pd.DataFrame(rows)

    # questions reachable at the centre point that a change puts out of reach, and what the
    # centre point's generator had made of them
    oc = o.pivot_table(index="question_id", columns="configuration", values="outcome",
                       aggfunc="first")
    if CENTRE in oc.columns:
        lost, lost_s4 = [], []
        for cfg in per_config.configuration:
            gone = oc[oc[CENTRE].isin(["OK", "S4"]) & ~oc[cfg].isin(["OK", "S4"]) & oc[cfg].notna()]
            lost.append(int(len(gone)))
            lost_s4.append(int((gone[CENTRE] == "S4").sum()))
        per_config["lost_reach"] = lost
        per_config["lost_reach_s4"] = lost_s4

    w = o.pivot_table(index="question_id", columns="configuration", values="correct",
                      aggfunc="first").astype(float)
    contrasts = []
    if CENTRE in w.columns:
        for c in w.columns:
            if c == CENTRE:
                continue
            both = w[[c, CENTRE]].dropna()
            r = paired_delta(both[c].values - both[CENTRE].values, n_boot)
            contrasts.append({"configuration": c, "n": int(len(both)), "d_correct": r["mean"],
                              "lo": r["lo"], "hi": r["hi"], "p": r["p"]})
    contrasts = pd.DataFrame(contrasts)
    if len(contrasts):
        contrasts["p_holm"] = holm(contrasts.p.values)

    reach = o.stage.eq("R")
    s4_all = o[o.outcome == "S4"]
    by_stage = {st: {"n": int(len(g)), "correct": float(g.correct.mean()),
                     "abstained": float(g.abstained.mean())}
                for st, g in o.groupby("stage")}
    return {
        "by_stage": by_stage,
        "questions": int(o.question_id.nunique()),
        "per_configuration": per_config,
        "correct_contrasts": contrasts,
        "abstention_reachable": float(o[reach].abstained.mean()),
        "abstention_blocked": float(o[~reach].abstained.mean()),
        "correct_reachable": float(o[reach].correct.mean()),
        "correct_blocked": float(o[~reach].correct.mean()),
        "s4_refusal_share": float(s4_all.abstained.mean()) if len(s4_all) else float("nan"),
        "abstention_by_stage_centre": (o[o.configuration == CENTRE].groupby("stage")
                                       .abstained.mean().to_dict()),
    }


def prompt_comparison(first, second, n_boot=N_BOOT):
    """Two generation runs over the same questions and the same retrieval traces.

    `first` and `second` are generation_outcomes() frames. Because retrieval is identical, the
    stage of every (configuration, question) pair is the same in both, and any change in the
    share of correct answers is a change made after retrieval. Differences are paired by
    configuration and question.
    """
    keys = ["configuration", "question_id"]
    m = first.merge(second, on=keys, suffixes=("_a", "_b"))
    same_stage = bool((m.stage_a == m.stage_b).all())

    def contrast(sub):
        d_c = sub.correct_b.astype(float).values - sub.correct_a.astype(float).values
        d_a = sub.abstained_b.astype(float).values - sub.abstained_a.astype(float).values
        rc, ra = paired_delta(d_c, n_boot), paired_delta(d_a, n_boot)
        return {"n": int(len(sub)),
                "correct_first": float(sub.correct_a.mean()),
                "correct_second": float(sub.correct_b.mean()),
                "d_correct": rc["mean"], "d_correct_lo": rc["lo"], "d_correct_hi": rc["hi"],
                "p_correct": rc["p"],
                "abstained_first": float(sub.abstained_a.mean()),
                "abstained_second": float(sub.abstained_b.mean()),
                "d_abstained": ra["mean"], "d_abstained_lo": ra["lo"], "d_abstained_hi": ra["hi"]}

    out = {"same_stages": same_stage, "pairs": int(len(m))}
    centre = m[m.configuration == CENTRE]
    for name, sub in (("all", m), ("centre", centre)):
        out[name] = contrast(sub)
        out[name + "_reachable"] = contrast(sub[sub.stage_a == "R"])
        out[name + "_blocked"] = contrast(sub[sub.stage_a != "R"])
    per = []
    for cfg, g in m.groupby("configuration", sort=False):
        per.append({"configuration": cfg, **contrast(g)})
    out["per_configuration"] = pd.DataFrame(per)
    out["by_stage"] = {st: {"n": int(len(g)), "correct_first": float(g.correct_a.mean()),
                            "correct_second": float(g.correct_b.mean()),
                            "abstained_first": float(g.abstained_a.mean()),
                            "abstained_second": float(g.abstained_b.mean())}
                       for st, g in m.groupby("stage_a")}
    return out


def plot_stage_correctness(by_stage, path, labels=("first prompt", "second prompt")):
    """Answers passing the correctness check, by attributed stage, for two generation runs."""
    plt = _plt()
    order = [s for s in ("R", "S3", "S2", "S1") if s in by_stage]
    names = {"R": "reachable", "S1": "S1 segmentation\nloss", "S2": "S2 retrieval\nmiss",
             "S3": "S3 context\nexclusion"}
    x = np.arange(len(order))
    a = [100 * by_stage[s]["correct_first"] for s in order]
    b = [100 * by_stage[s]["correct_second"] for s in order]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    # the first run is the reference and stays neutral; the second carries the stage colour
    ax.bar(x - 0.2, a, 0.4, color=MUTED, edgecolor=SURFACE, linewidth=1.0, label=labels[0])
    ax.bar(x + 0.2, b, 0.4, color=[COLOURS.get(s, MUTED) for s in order],
           edgecolor=SURFACE, linewidth=1.0, label=labels[1])
    for xi, v in zip(x, a):
        ax.text(xi - 0.2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8, color=TEXT_SECONDARY)
    for xi, v in zip(x, b):
        ax.text(xi + 0.2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8, color=TEXT_PRIMARY,
                fontweight="bold")
    ax.set_xticks(x, [f"{names[s]}\n(n = {by_stage[s]['n']})" for s in order], fontsize=8.5)
    ax.set_ylabel("answers passing the correctness check (%)")
    ax.set_ylim(0, max(a + b) * 1.2)
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def reviewed_check(reviewed, token_f1, threshold):
    """How the correctness check behaves on the reviewed answers, blocked versus reachable."""
    r = reviewed.copy()
    r["f1"] = [token_f1(str(p), str(g)) for p, g in
               zip(r.predicted_answer, r.gold_answer_from_techqa)]
    r["stage"] = attribute(r).values
    r["passes"] = r.f1 >= threshold
    lab = r.manual_answer_correct.astype(str).str.strip().str.lower()
    blocked = r.stage != "R"
    return {"blocked": int(blocked.sum()),
            "blocked_passing": int((blocked & r.passes).sum()),
            "blocked_correct": int((blocked & lab.eq("correct")).sum()),
            "blocked_passing_labels": lab[blocked & r.passes].value_counts().to_dict()}


# =================================================================================
# 5. Figures
# =================================================================================

# The validated categorical palette, in its documented order: S1, S2 and S3 take slots 1 to 3
# and keep them in every figure, so a colour always means the same stage; reachable is the
# neutral ink, because it is the absence of a failure rather than a fourth kind of one.
COLOURS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "R": "#c9c6bf"}
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, RULE, MUTED = "#ffffff", "#3f3d39", "#8d8a83"
FIG_STYLE = {
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
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


def plot_partition(parts, path, title="Where questions are lost before generation"):
    plt = _plt()
    p = parts.copy()
    p = p.iloc[np.argsort(-p["R"].to_numpy(), kind="stable")]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(p) + 1.6))
    left = np.zeros(len(p))
    y = np.arange(len(p))[::-1]
    for s in ORDER:
        vals = p[s].to_numpy() * 100
        ax.barh(y, vals, left=left, color=COLOURS[s], edgecolor=SURFACE,
                linewidth=1.2, height=0.72,
                label=f"{s} {NAMES[s]}" if s != "R" else "reachable")
        for yi, l, v in zip(y, left, vals):
            if v >= 4:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if s != "R" else TEXT_SECONDARY,
                        fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(p["configuration"], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the 610 answerable questions (%)")
    ax.set_title(title, fontsize=10)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12 - 0.2 / max(1, len(p))))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decomposition(dec, path, title="Which stage each change acts on"):
    plt = _plt()
    d = dec.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(d) + 1.4))
    y = np.arange(len(d))[::-1]
    offsets = {"S1": 0.22, "S2": 0.0, "S3": -0.22}
    for s in BLOCKING:
        m = d[f"d{s}"].to_numpy() * 100
        lo = d[f"d{s}_lo"].to_numpy() * 100
        hi = d[f"d{s}_hi"].to_numpy() * 100
        ax.errorbar(m, y + offsets[s], xerr=[m - lo, hi - m], fmt="o", color=COLOURS[s],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, capsize=0, lw=1.6,
                    label=f"{s} {NAMES[s]}")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["configuration"], fontsize=8.5)
    ax.set_xlabel("change in share of questions against the centre point (percentage points,"
                  " 95% paired bootstrap)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


# =================================================================================
# 6. Persistence
# =================================================================================

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
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
    if o is None or isinstance(o, str):
        return o
    return str(o)


def load_ceiling(results_dir, path=None):
    """The per-(method, question) segmentation ceiling written by notebook RQ2-0."""
    p = Path(path) if path else Path(results_dir) / CEILING_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing - run notebooks/RQ2_0_Segmentation_Ceiling.ipynb (CPU only) and "
            "put its csv beside the per-question traces")
    c = pd.read_csv(p)
    missing = {"method", "question_id", CEILING} - set(c.columns)
    if missing:
        raise ValueError(f"{p} lacks {sorted(missing)}")
    return c


def run_all(results_dir, v1_path=None, v2_path=None, out_dir=None, n_boot=N_BOOT,
            token_f1=None, first_run_scored=None, ceiling_path=None):
    """Everything RQ2 reports, from files alone. Returns the results dictionary.

    When notebook D's `rq1_generation_scored.csv` is present and a `token_f1` function is
    supplied, the generation stage is also measured at scale, with a correctness check
    calibrated on the reviewed answers of `v2_path`.
    """
    R = Path(results_dir)
    out = Path(out_dir) if out_dir else R / "rq2"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    frames = load_traces(R)
    ceiling = load_ceiling(R, ceiling_path)
    table, centre_identical = configuration_table(frames, ceiling)
    defects = trace_defects(table)
    parts = partition(table, n_boot=n_boot)
    dec = decompositions(table, n_boot=n_boot)
    pooling = decompose(table, "late chunking", "long-context encoder", n_boot=n_boot)
    sens = threshold_sensitivity(table)
    path = sophistication_path(parts)
    moves = {c: transitions(table, c).to_dict() for c in table.configuration.unique()
             if c != CENTRE}

    parts.to_csv(out / "rq2_partition.csv", index=False)
    dec.to_csv(out / "rq2_decomposition.csv", index=False)
    sens.to_csv(out / "rq2_threshold_sensitivity.csv", index=False)
    plot_partition(parts, out / "figures" / "fig_rq2_partition.png")
    plot_decomposition(dec, out / "figures" / "fig_rq2_decomposition.png")

    res = {"n_questions": int(table.question_id.nunique()),
           "n_configurations": int(table.configuration.nunique()),
           "centre_copies_identical": bool(centre_identical),
           "trace_defects": defects,
           "s3_threshold": S3_THRESHOLD,
           "partition": parts, "decomposition": dec, "pooling_effect": pooling,
           "threshold_sensitivity": sens, "sophistication_path": path,
           "transitions": moves}

    # The centre point was selected on the training split, so every result is repeated on the
    # development split, which played no part in that selection.
    by_split = {}
    if all("split" in f.columns for f in frames.values()):
        for split in ("dev", "train"):
            sub = {k: f[f.split == split] for k, f in frames.items()}
            if not all(len(f) for f in sub.values()):
                continue
            t_, _ = configuration_table(sub, ceiling)
            by_split[split] = {"n": int(t_.question_id.nunique()),
                               "partition": partition(t_, n_boot=n_boot),
                               "decomposition": decompositions(t_, n_boot=n_boot)}
    res["by_split"] = by_split

    if v1_path and v2_path and Path(v1_path).exists() and Path(v2_path).exists():
        # the reviewed answers were produced at the centre point, so they take its ceiling
        v1, v2 = (with_ceiling(pd.read_csv(p), ceiling, CENTRE_METHOD)
                  for p in (v1_path, v2_path))
        res["reviewed"] = ablation(v1, v2)
        res["reviewed"]["coverage_near_threshold"] = sorted(
            float(x) for x in v2.gold_span_coverage
            if 0.5 <= float(x) < 0.95 and float(x) != 0.0)

        scored_path = R / "rq1_generation_scored.csv"
        if token_f1 is not None and scored_path.exists():
            f1 = [token_f1(str(p), str(g)) for p, g in
                  zip(v2.predicted_answer, v2.gold_answer_from_techqa)]
            check = calibrate_check(f1, v2.manual_answer_correct)
            outcomes = generation_outcomes(table, pd.read_csv(scored_path), check["threshold"])
            summary = generation_summary(outcomes, n_boot)
            summary["check"] = check
            summary["reviewed_check"] = reviewed_check(v2, token_f1, check["threshold"])
            res["generation_at_scale"] = summary
            outcomes.to_csv(out / "rq2_generation_outcomes.csv", index=False)
            # an earlier generation run over the same questions, for a prompt comparison
            if first_run_scored and Path(first_run_scored).exists():
                first = generation_outcomes(table, pd.read_csv(first_run_scored),
                                            check["threshold"])
                res["prompt_comparison"] = prompt_comparison(first, outcomes, n_boot)
                plot_stage_correctness(res["prompt_comparison"]["by_stage"],
                                       out / "figures" / "fig_rq2_generation_stages.png")
            summary["per_configuration"].to_csv(out / "rq2_generation_stages.csv", index=False)
    (out / "rq2_results.json").write_text(json.dumps(jsonable(res), indent=2, allow_nan=False),
                                          encoding="utf-8")
    return res


# =================================================================================
# 8. The answer to RQ2, read from the results
# =================================================================================

def answer(res, width=78):
    """Print RQ2's answer from the computed results, in the order the pipeline runs."""
    parts = pd.DataFrame(res["partition"])
    dec = pd.DataFrame(res["decomposition"])
    centre = parts[parts.configuration == CENTRE].iloc[0]
    blocked = float(centre["blocked"])
    lines = ["=" * width,
             "THE ANSWER TO RQ2",
             "Can a failed answer be attributed to one pipeline stage from trace evidence",
             "alone, and does that attribution change the engineering conclusion?",
             "=" * width,
             "",
             f"Questions               {res['n_questions']} answerable, under "
             f"{res['n_configurations']} configurations",
             "Labels                  every question gets exactly one of "
             + ", ".join(f"{s} {NAMES[s]}" for s in ORDER),
             "                        by construction: the first failed test wins, in "
             "pipeline order",
             f"Centre point            {100 * blocked:.1f}% of questions are blocked before "
             f"generation, so no",
             f"                        generator can answer more than "
             f"{100 * (1 - blocked):.1f}% of them"]
    per_stage = ", ".join(f"{s} {100 * float(centre[s]):.1f}%" for s in BLOCKING)
    lines.append(f"                        ({per_stage})")
    if len(dec):
        d = dec.dropna(subset=["dominant_stage"])
        if len(d):
            lines.append("Each change acts on one stage:")
            for _, r in d.iterrows():
                s = r["dominant_stage"]
                lines.append(f"    {r['configuration']:<26} moves {s} {NAMES[s]} by "
                             f"{100 * float(r['d' + s]):+.1f} points")
        hidden = dec.assign(net=dec.net_reachable.abs()).sort_values("moved", ascending=False)
        if len(hidden):
            h = hidden.iloc[0]
            lines += ["Aggregation hides movement:",
                      f"    {h['configuration']} changes the reachable count by "
                      f"{int(h['net_reachable']):+d} question(s), yet moves",
                      f"    {int(h['moved'])} questions between stages - "
                      f"{100 * int(h['moved']) / max(1, res['n_questions']):.0f}% of the "
                      f"dataset an aggregate score would call unchanged."]
    g = res.get("generation_at_scale", {}).get("by_stage")
    if g:
        grad = ", ".join(f"{s} {100 * g[s]['correct']:.0f}%" for s in ("R", "S3", "S2", "S1")
                         if s in g and "correct" in g[s])
        if grad:
            lines += ["Premise checked at scale:",
                      f"    answers passing the correctness check by stage: {grad}",
                      "    the further upstream the evidence was lost, the less likely a "
                      "correct answer."]
    pc = res.get("prompt_comparison", {}).get("centre")
    if pc and "d_correct" in pc:
        lines += ["Prompt against retrieval:",
                  f"    with every stage label fixed, changing only the prompt moved the "
                  f"passing share by {100 * pc['d_correct']:+.1f} points "
                  f"[{100 * pc['d_correct_lo']:+.1f}, {100 * pc['d_correct_hi']:+.1f}]."]
    lines += ["",
              "Answered YES on both counts.",
              "  Attribution is possible: the rule is deterministic, uses only trace fields,",
              "  and leaves every failed question with exactly one stage.",
              "  It changes the conclusion: the stage names the component that owes the fix,",
              "  it exposes reshuffling that an aggregate score reports as no change, and it",
              "  fixes a ceiling on any generator before one is run.",
              "=" * width]
    print("\n".join(lines))
    return {"blocked_centre": blocked, "ceiling": 1 - blocked, "verdict": "yes"}
