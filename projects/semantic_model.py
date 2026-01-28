"""
Demo: 7-category weekly review pipeline + report
- Clean → negative filter (rating<=3) → WoW keys
- Exploratory clustering on negative reviews (SBERT → HDBSCAN)
- Weak-supervision tagging (Matcher-like rules) → unmatched
- Cluster unmatched negatives → refine rules (demo) → re-run tagging
- Generate:
  (1) Category Health (Negative Issue Share, WoW)
  (2) Top Emerging Issues (7d vs 28d baseline) from clusters
  (3) Representative Evidence
  (4) Output "data warehouse" table

Install (recommended):
  pip install -U pandas numpy sentence-transformers umap-learn hdbscan scikit-learn

Note:
- Using a multilingual SBERT model to handle Chinese + English reviews.
- This is a small sample input; scale-up is the same code path.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- SBERT + clustering deps ---
from sentence_transformers import SentenceTransformer
import umap
import hdbscan


# ----------------------------
# 0) Sample Input (toy data)
# ----------------------------
def sample_input() -> pd.DataFrame:
    # Two windows:
    # - baseline window: 28d total
    # - recent window: last 7d
    today = pd.Timestamp("2026-01-27")  # consistent with your timezone note
    def d(days_ago: int) -> pd.Timestamp:
        return today - pd.Timedelta(days=days_ago)

    rows = [
        # TECH (recent 7d): "overheat"
        dict(product_id="p101", product_name="Charger 65W", product_category="tech",
             product_sub_category="charger", review_id="r001", review_time=d(2),
             review_content="充电器温度很高，摸着烫手，有点担心安全。", rating=2),
        dict(product_id="p102", product_name="Earbuds X", product_category="tech",
             product_sub_category="earbuds", review_id="r002", review_time=d(3),
             review_content="耳机戴一会儿就发烫，电量掉得也快。", rating=3),
        dict(product_id="p102", product_name="Earbuds X", product_category="tech",
             product_sub_category="earbuds", review_id="r003", review_time=d(4),
             review_content="发热明显，续航也不行。", rating=2),
        # TECH baseline (older): other issues
        dict(product_id="p103", product_name="Fan Mini", product_category="tech",
             product_sub_category="fan", review_id="r004", review_time=d(18),
             review_content="噪音有点大，但还行。", rating=3),
        dict(product_id="p103", product_name="Fan Mini", product_category="tech",
             product_sub_category="fan", review_id="r005", review_time=d(25),
             review_content="风力太弱，不如预期。", rating=3),

        # HOME (recent): package wreck
        dict(product_id="p201", product_name="Mug", product_category="home",
             product_sub_category="cups", review_id="r101", review_time=d(1),
             review_content="包装破损，杯子到手就裂了，物流太暴力。", rating=2),
        dict(product_id="p202", product_name="Bowl Set", product_category="home",
             product_sub_category="kitchen", review_id="r102", review_time=d(5),
             review_content="外箱压扁，里面碎了一只碗。", rating=2),
        # HOME baseline: smell/material
        dict(product_id="p203", product_name="Bedsheet", product_category="home",
             product_sub_category="linens", review_id="r103", review_time=d(20),
             review_content="有刺鼻气味，洗了两次还有味道。", rating=2),

        # CLOTHING (recent): size unfit
        dict(product_id="p301", product_name="Jacket", product_category="clothing",
             product_sub_category="outerwear", review_id="r201", review_time=d(2),
             review_content="尺码明显偏小，穿着很紧，活动不方便。", rating=2),
        dict(product_id="p302", product_name="Pants", product_category="clothing",
             product_sub_category="pants", review_id="r202", review_time=d(6),
             review_content="版型怪，腰合适但腿很紧。", rating=3),
        # CLOTHING baseline: color mismatch
        dict(product_id="p303", product_name="Sweater", product_category="clothing",
             product_sub_category="tops", review_id="r203", review_time=d(16),
             review_content="颜色和图片差很多，有点失望。", rating=3),

        # FOOD (mix)
        dict(product_id="p401", product_name="Snack A", product_category="food",
             product_sub_category="snacks", review_id="r301", review_time=d(3),
             review_content="太甜了，齁嗓子。", rating=2),
        dict(product_id="p401", product_name="Snack A", product_category="food",
             product_sub_category="snacks", review_id="r302", review_time=d(22),
             review_content="味道还行，但是分量有点少。", rating=3),

        # WINE
        dict(product_id="p501", product_name="Wine Red", product_category="wine",
             product_sub_category="red", review_id="r401", review_time=d(4),
             review_content="瓶口有渗漏，包装也一般。", rating=2),
        dict(product_id="p502", product_name="Wine White", product_category="wine",
             product_sub_category="white", review_id="r402", review_time=d(17),
             review_content="口感不如描述，偏酸。", rating=3),

        # OUTDOOR
        dict(product_id="p601", product_name="Camp Chair", product_category="outdoor",
             product_sub_category="chairs", review_id="r501", review_time=d(5),
             review_content="椅子用了两次就断了，感觉材料不行。", rating=2),
        dict(product_id="p602", product_name="Sleeping Bag", product_category="outdoor",
             product_sub_category="sleeping_bags", review_id="r502", review_time=d(23),
             review_content="不够保暖，宣传有点夸张。", rating=3),

        # SPORTS
        dict(product_id="p701", product_name="Knee Pad", product_category="sports",
             product_sub_category="protective", review_id="r601", review_time=d(6),
             review_content="穿戴不舒服，勒得慌。", rating=3),
        dict(product_id="p702", product_name="Yoga Mat", product_category="sports",
             product_sub_category="mats", review_id="r602", review_time=d(21),
             review_content="有点滑，不太防滑。", rating=3),

        # Some empty/dup/noise to show cleaning
        dict(product_id="p999", product_name="NA", product_category="food",
             product_sub_category="snacks", review_id="r900", review_time=d(2),
             review_content="  ", rating=1),
        dict(product_id="p401", product_name="Snack A", product_category="food",
             product_sub_category="snacks", review_id="r901", review_time=d(3),
             review_content="太甜了，齁嗓子。", rating=2),  # duplicate content
        dict(product_id="p888", product_name="NA", product_category="tech",
             product_sub_category="fan", review_id="r902", review_time=d(3),
             review_content="差", rating=2),  # very short (noise)
    ]
    return pd.DataFrame(rows)


# ----------------------------
# 1) Clean + Negative Filter + Time Keys
# ----------------------------
def clean_reviews(df: pd.DataFrame,
                  text_col="review_content",
                  min_chars=6) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    df = df[df[text_col].str.len() >= min_chars]
    df = df.drop_duplicates(subset=[text_col, "product_id", "review_time"]).reset_index(drop=True)
    return df


def add_time_keys(df: pd.DataFrame, time_col="review_time") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    # ISO week label (WoW)
    iso = df[time_col].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    df["week_key"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
    return df


def filter_negative(df: pd.DataFrame, rating_col="rating", threshold=3) -> pd.DataFrame:
    df = df.copy()
    df["if_negative"] = df[rating_col] <= threshold
    df["sentiment"] = np.where(df["if_negative"], "negative", "non_negative")
    return df[df["if_negative"]].reset_index(drop=True)


# ----------------------------
# 2) Exploratory clustering (SBERT → UMAP → HDBSCAN)
# ----------------------------
@dataclass
class ClusterConfig:
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.05
    hdb_min_cluster_size: int = 2
    hdb_min_samples: int = 1
    seed: int = 42


def cluster_texts_sbert_hdbscan(texts: List[str], cfg: ClusterConfig):
    model = SentenceTransformer(cfg.sbert_model)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed
    )
    emb_2d = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        min_samples=cfg.hdb_min_samples,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(emb_2d)

    return emb, emb_2d, labels


# ----------------------------
# 3) Weak-supervision tagging (Matcher-like rules)
#    (In your real project: spaCy Matcher rules.)
# ----------------------------
# Minimal rule sets for demo (category -> sub_aspect -> keyword patterns)
RULES: Dict[str, Dict[str, List[str]]] = {
    "tech": {
        "overheat_heat_control": ["发烫", "烫手", "温度很高", "发热", "过热", "overheat", "hot"],
        "battery_drain": ["电量掉", "续航", "battery"],
        "noise": ["噪音", "loud"],
    },
    "home": {
        "package_wreck": ["包装破损", "压扁", "碎", "裂", "破了", "damaged"],
        "odor_smell": ["刺鼻", "气味", "味道很大", "smell"],
    },
    "clothing": {
        "size_unfit": ["尺码", "偏小", "偏大", "很紧", "不合身", "版型", "腰合适但腿很紧"],
        "limited_movement": ["活动不方便", "不好动", "不灵活", "restrictive", "hard to move"],
        "color_mismatch": ["颜色", "图片差", "色差"],
    },
    "food": {
        "too_sweet": ["太甜", "齁", "sweet"],
        "portion_small": ["分量少", "太少"],
    },
    "wine": {
        "leak_damage": ["渗漏", "漏", "破损", "碎"],
        "taste_not_expected": ["偏酸", "不如描述", "口感"],
    },
    "outdoor": {
        "durability_breakage": ["断了", "坏了", "不结实", "材料不行", "broke"],
        "warmth_insulation": ["不够保暖", "不暖和"],
    },
    "sports": {
        "comfort_fit": ["不舒服", "勒", "紧", "fit"],
        "slippery_grip": ["滑", "不防滑", "slip"],
    }
}

# Map sub_aspect to top-level aspect (for report)
SUB_TO_ASPECT: Dict[str, str] = {
    "overheat_heat_control": "overheat/heat control",
    "package_wreck": "package wreck",
    "size_unfit": "size unfit",
    # You can extend…
}


def tag_with_rules(category: str, text: str) -> Tuple[List[str], List[str]]:
    """
    Returns:
      complain_aspects: list[str]
      complain_sub_aspect_list: list[str]
    """
    sub_hits = []
    rules = RULES.get(category, {})
    t = text.lower()

    for sub, pats in rules.items():
        if any(p.lower() in t for p in pats):
            sub_hits.append(sub)

    # Map subs -> aspects (dedupe)
    aspects = []
    for sub in sub_hits:
        aspects.append(SUB_TO_ASPECT.get(sub, sub))  # fallback: sub itself
    aspects = sorted(set(aspects))
    return aspects, sub_hits


# ----------------------------
# 4) Emerging issue scoring (7d vs 28d baseline)
# ----------------------------
def emerging_score(c7: int, cbase: int, sku7: int) -> float:
    """
    Simple interpretable score:
      lift = (c7 + 1) / (cbase + 1)
      score = lift * log1p(c7) * log1p(sku7)
    """
    lift = (c7 + 1) / (cbase + 1)
    return float(lift * math.log1p(c7) * math.log1p(sku7))


def build_report(df_all: pd.DataFrame, df_neg: pd.DataFrame,
                 labels: np.ndarray,
                 recent_days=7, baseline_days=28):
    today = df_all["review_time"].max().normalize()
    recent_start = today - pd.Timedelta(days=recent_days)
    base_start = today - pd.Timedelta(days=baseline_days)

    # (1) Category Health: Negative issue share + WoW pp
    # We'll compute current week and previous week based on week_key.
    tmp_all = add_time_keys(df_all)
    tmp_neg = add_time_keys(df_neg)

    # Negative share by category in latest week
    last_week = tmp_all["week_key"].max()
    prev_week = (tmp_all.sort_values(["iso_year", "iso_week"])
                        .drop_duplicates("week_key")["week_key"].iloc[-2]
                 if tmp_all["week_key"].nunique() >= 2 else last_week)

    def neg_share_for_week(week_key: str) -> pd.Series:
        total = tmp_all[tmp_all["week_key"] == week_key].groupby("product_category")["review_id"].count()
        neg = tmp_neg[tmp_neg["week_key"] == week_key].groupby("product_category")["review_id"].count()
        share = (neg / total).fillna(0.0)
        return share

    share_last = neg_share_for_week(last_week)
    share_prev = neg_share_for_week(prev_week)
    wow_pp = (share_last - share_prev) * 100.0

    categories = ["tech", "food", "clothing", "home", "outdoor", "wine", "sports"]

    # (2) Emerging issues: based on clusters within negatives
    neg = df_neg.copy()
    neg["cluster"] = labels

    # define windows
    in_recent = (neg["review_time"] >= recent_start) & (neg["review_time"] <= today + pd.Timedelta(days=1))
    in_base = (neg["review_time"] >= base_start) & (neg["review_time"] < recent_start)

    # Compute per (category, cluster): recent count, base count, sku coverage, example texts
    emerg_rows = []
    for (cat, cl), g in neg.groupby(["product_category", "cluster"]):
        if cl == -1:
            continue
        c7 = int((g["review_time"] >= recent_start).sum())
        cbase = int(((g["review_time"] >= base_start) & (g["review_time"] < recent_start)).sum())
        sku7 = g.loc[g["review_time"] >= recent_start, "product_id"].nunique()

        # need enough recent signal
        if c7 < 2:
            continue

        score = emerging_score(c7, cbase, sku7)
        lift = (c7 + 1) / (cbase + 1)

        # label cluster by rule-based sub_aspect majority (for demo)
        # In real project: you would manually name clusters or use keyword extraction.
        sub_counts = {}
        for txt in g.loc[g["review_time"] >= recent_start, "review_content"].tolist():
            _, subs = tag_with_rules(cat, txt)
            for s in subs:
                sub_counts[s] = sub_counts.get(s, 0) + 1
        issue = max(sub_counts, key=sub_counts.get) if sub_counts else f"cluster_{cl}"

        emerg_rows.append(dict(
            category=cat,
            issue=SUB_TO_ASPECT.get(issue, issue),
            score=score,
            lift=lift,
            sku_cnt=int(g["product_id"].nunique()),
            cluster=int(cl),
        ))

    emerg = pd.DataFrame(emerg_rows).sort_values("score", ascending=False).head(3)

    # (3) Representative evidence: pick 2 examples from top emerging cluster(s)
    evidence = []
    for _, row in emerg.iterrows():
        cat = row["category"]
        cl = row["cluster"]
        ex = neg[(neg["product_category"] == cat) & (neg["cluster"] == cl)].sort_values("review_time", ascending=False).head(2)
        for _, r in ex.iterrows():
            evidence.append(dict(
                category=cat,
                issue=row["issue"],
                text=r["review_content"],
                sku=r["product_id"],
            ))

    # Print report (compact)
    def bar(pct: float) -> str:
        # 8-block bar
        filled = int(round(pct / 100 * 8))
        filled = max(0, min(8, filled))
        return "■" * filled + "□" * (8 - filled)

    print("📊 Weekly Review Signal Snapshot (7 Categories)")
    print("1) Category Health (Negative Issue Share, WoW)")
    for cat in categories:
        pct = float(share_last.get(cat, 0.0) * 100.0)
        pp = float(wow_pp.get(cat, 0.0))
        arrow = "↑" if pp > 0.01 else ("↓" if pp < -0.01 else "→")
        print(f"• {cat.capitalize():<8}: {bar(pct)} {pct:.0f}% ({arrow} {pp:+.0f}pp)")

    print("\n2) Top Emerging Issues (7d vs 28d baseline)")
    if emerg.empty:
        print("  (no emerging clusters met threshold)")
    else:
        for i, r in enumerate(emerg.itertuples(index=False), 1):
            print(f"{i}. {r.category.capitalize()} | {r.issue} — Score {r.score:.1f} (↑ {r.lift:.1f}×) — SKUs {r.sku_cnt}")

    print("\n3) Representative Evidence (Drill-down)")
    for e in evidence:
        print(f"• {e['category'].capitalize()} | {e['issue']}")
        print(f"    ◦ “{e['text']}” ({e['sku']})")

    print("\n4) Action Suggestions (Ops-ready)")
    print("• Tech 过热：按供应商批次对比退货率，优先排查近 7 天新增 SKU。")
    print("• 包装破损：按承运商 × 仓库拆分破损率，先处理 Top 3 组合。")
    print("• 服装尺码：对高覆盖 SKU 复核尺码表与退换原因。")

    return emerg, pd.DataFrame(evidence)


# ----------------------------
# 5) Build "data warehouse" output table
# ----------------------------
def build_dw_table(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df = add_time_keys(df)
    df["if_negative"] = df["rating"] <= 3
    df["sentiment"] = np.where(df["if_negative"], "negative", "non_negative")

    # Tag only negatives for complaint aspects
    aspects_list = []
    sub_list = []
    for r in df.itertuples(index=False):
        if r.if_negative:
            aspects, subs = tag_with_rules(r.product_category, r.review_content)
        else:
            aspects, subs = [], []
        aspects_list.append(aspects)
        sub_list.append(subs)

    df["complain_aspects"] = aspects_list
    df["complain_sub_aspect_list"] = sub_list

    # reorder columns as requested
    cols = [
        "product_id", "product_name", "product_category", "product_sub_category",
        "review_id", "review_time", "review_content",
        "sentiment", "if_negative",
        "complain_aspects", "complain_sub_aspect_list"
    ]
    return df[cols].sort_values(["product_category", "review_time"]).reset_index(drop=True)


# ----------------------------
# 6) Main demo run
# ----------------------------
def main():
    # Read data
    df_raw = sample_input()
    print("Raw rows:", len(df_raw))

    # Clean data
    df_clean = clean_reviews(df_raw, min_chars=6)
    print("After cleaning:", len(df_clean))

    # Time keys for WoW
    df_clean = add_time_keys(df_clean)

    # Negative filter (rating <= 3)
    df_neg = filter_negative(df_clean, threshold=3)
    print("Negative reviews:", len(df_neg))

    # Exploratory clustering on negative reviews (SBERT → HDBSCAN)
    cfg = ClusterConfig(hdb_min_cluster_size=2, hdb_min_samples=1)  # small because demo data is tiny
    texts = df_neg["review_content"].tolist()
    _, _, labels = cluster_texts_sbert_hdbscan(texts, cfg)

    # Weak-supervision tagging (Matcher-like) and unmatched set
    tagged = []
    unmatched = []
    for t in df_neg["review_content"].tolist():
        # category-specific rules require category; we'll tag later in DW table
        tagged.append(True)  # placeholder; we compute true tagging below

    # "Unmatched" here means: no sub_aspects hit by rules
    df_neg2 = df_neg.copy()
    aspects_col = []
    subs_col = []
    for r in df_neg2.itertuples(index=False):
        aspects, subs = tag_with_rules(r.product_category, r.review_content)
        aspects_col.append(aspects)
        subs_col.append(subs)
    df_neg2["complain_aspects"] = aspects_col
    df_neg2["complain_sub_aspect_list"] = subs_col
    df_neg2["unmatched"] = df_neg2["complain_sub_aspect_list"].apply(lambda x: len(x) == 0)

    print("Tagged negatives:", int((~df_neg2["unmatched"]).sum()))
    print("Unmatched negatives:", int(df_neg2["unmatched"].sum()))

    # Cluster unmatched negatives again (diagnostic)
    # (demo: we just show how; in real data you'd get meaningful clusters)
    df_unmatched = df_neg2[df_neg2["unmatched"]].reset_index(drop=True)
    if len(df_unmatched) >= 5:  # need enough data
        u_texts = df_unmatched["review_content"].tolist()
        _, _, u_labels = cluster_texts_sbert_hdbscan(u_texts, cfg)
        df_unmatched["unmatched_cluster"] = u_labels
        # Here you'd inspect clusters and update RULES accordingly.
        # Demo "update": add a new rule under food for "too salty" if found.
        RULES["food"].setdefault("too_salty", [])
        RULES["food"]["too_salty"] += ["太咸", "咸"]
        print("Rule updated: food.too_salty += ['太咸','咸']")
    else:
        print("Unmatched set too small in demo; skipping second clustering.")

    # Re-run tagging after rule update (coverage tracking)
    aspects_col2 = []
    subs_col2 = []
    for r in df_neg2.itertuples(index=False):
        aspects, subs = tag_with_rules(r.product_category, r.review_content)
        aspects_col2.append(aspects)
        subs_col2.append(subs)
    df_neg2["complain_aspects_v2"] = aspects_col2
    df_neg2["complain_sub_aspect_list_v2"] = subs_col2
    df_neg2["unmatched_v2"] = df_neg2["complain_sub_aspect_list_v2"].apply(lambda x: len(x) == 0)
    print("Unmatched negatives after update:", int(df_neg2["unmatched_v2"].sum()))

    # Generate report (based on clustering labels + WoW keys)
    build_report(df_clean, df_neg, labels)

    # Build DW output table
    df_dw = build_dw_table(df_clean)
    print("\n--- Data Warehouse Table (sample) ---")
    print(df_dw.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
