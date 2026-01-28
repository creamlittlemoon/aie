"""
Demo: Car optional-config recommendation (Supervised + CF), end-to-end
Assumptions aligned with your constraints:
- Inputs: wishlist (must-choose context) + final order (observed outcomes)
- No offer/quote logs => labels represent "final-order likelihood", not true acceptance
- Mixed optional types: some binary, some multiclass
- No hierarchy
- Per car type split (demo uses ET5)
- Supervised models provide personalized probabilities conditioned on must-choose context
- Item-based CF provides bundle/co-purchase hints (supporting signal)

This script prints SAMPLE INPUT + SAMPLE OUTPUT for each step.
Dependencies: pandas, numpy, scikit-learn
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# STEP 1) Sample input data (wishlist + order)
# =========================================================
wishlist = pd.DataFrame([
    # must-choose configs only
    {"user_id":"U100","customer_name":"Alice","car_type":"ET5","battery":"100kWh","color":"Black","wheel":"21",
     "interior_material":"Leather","nomi":"Yes","key1_color":"Gold","wish_ts":"2025-12-01"},
    {"user_id":"U101","customer_name":"Ben","car_type":"ET5","battery":"75kWh","color":"White","wheel":"19",
     "interior_material":"Fabric","nomi":"No","key1_color":"Black","wish_ts":"2025-12-02"},
    {"user_id":"U102","customer_name":"Cara","car_type":"ET5","battery":"100kWh","color":"Grey","wheel":"20",
     "interior_material":"Leather","nomi":"Yes","key1_color":"Black","wish_ts":"2025-12-03"},

    # new user to score (no order yet)
    {"user_id":"U200","customer_name":"Sylvia Zhang","car_type":"ET5","battery":"100kWh","color":"Black","wheel":"21",
     "interior_material":"Leather","nomi":"Yes","key1_color":"Gold","wish_ts":"2025-12-20"},
])

orders = pd.DataFrame([
    # optional outcomes: mixed binary + multiclass (flat, no hierarchy)
    {"order_id":"O1","user_id":"U100","car_type":"ET5","order_ts":"2025-12-05",
     "pilot":1, "fridge":0, "audio":"premium", "calipers":"green"},
    {"order_id":"O2","user_id":"U101","car_type":"ET5","order_ts":"2025-12-08",
     "pilot":0, "fridge":0, "audio":"standard", "calipers":"none"},
    {"order_id":"O3","user_id":"U102","car_type":"ET5","order_ts":"2025-12-10",
     "pilot":0, "fridge":1, "audio":"premium", "calipers":"orange"},
])

wishlist["wish_ts"] = pd.to_datetime(wishlist["wish_ts"])
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

print("\n==================== STEP 1 INPUT: wishlist ====================")
print(wishlist)
print("\n==================== STEP 1 INPUT: orders ======================")
print(orders)


# =========================================================
# STEP 2) Data cleaning & combine (align wishlist snapshot to order)
# (Demo simplification: 1 wishlist row per user; production: latest wish_ts <= order_ts)
# =========================================================
# In prod: standardize enums here (e.g., 'Blk'->'Black') and dedup orders/cancellations.

base = orders.merge(
    wishlist.drop(columns=["wish_ts"]),  # keep user/customer fields and must-choose
    on=["user_id","car_type"],
    how="inner"
)

print("\n==================== STEP 2 OUTPUT: aligned base ====================")
# This is the training base: must-context + observed outcomes
print(base[[
    "order_id","user_id","customer_name","car_type",
    "battery","color","wheel","interior_material","nomi","key1_color",
    "pilot","fridge","audio","calipers"
]])


# =========================================================
# STEP 3) Config taxonomy spec (binary vs multiclass) + missing rules
# =========================================================
cfg_dict = pd.DataFrame([
    {"config_key":"pilot","type":"binary","allowed_values":[0,1],"missing_policy":"null->0_or_unknown"},
    {"config_key":"fridge","type":"binary","allowed_values":[0,1],"missing_policy":"null->0_or_unknown"},
    {"config_key":"audio","type":"multiclass","allowed_values":["standard","premium"],"missing_policy":"null->unknown"},
    {"config_key":"calipers","type":"multiclass","allowed_values":["none","green","orange"],"missing_policy":"null->unknown"},
])

print("\n==================== STEP 3 OUTPUT: cfg_dict ====================")
print(cfg_dict)


# =========================================================
# STEP 4) Car-type split (each car type may support different configs/values)
# (Demo uses ET5 only)
# =========================================================
car_type = "ET5"
data = base[base["car_type"] == car_type].copy()

supported_configs = ["pilot","fridge","audio","calipers"]  # per car type in prod
print(f"\n==================== STEP 4 OUTPUT: {car_type} dataset ====================")
print("rows:", len(data))
print("supported configs:", supported_configs)


# =========================================================
# STEP 5) Convert raw data into model-ready targets (per optional config)
# - binary targets: y in {0,1}
# - multiclass targets: y in {v1,v2,...}
# =========================================================
must_cols = ["battery","color","wheel","interior_material","nomi","key1_color"]

# Example target datasets
ds_pilot = data[must_cols + ["pilot"]].rename(columns={"pilot":"y_pilot"})
ds_fridge = data[must_cols + ["fridge"]].rename(columns={"fridge":"y_fridge"})
ds_audio = data[must_cols + ["audio"]].rename(columns={"audio":"y_audio"})
ds_calipers = data[must_cols + ["calipers"]].rename(columns={"calipers":"y_calipers"})

print("\n==================== STEP 5 OUTPUT: ds_pilot (binary) ====================")
print(ds_pilot)
print("\n==================== STEP 5 OUTPUT: ds_audio (multiclass) ====================")
print(ds_audio)


# =========================================================
# STEP 6) Feature engineering (one-hot encoding)
# X is must-choose context only (no optional columns, no user_id)
# =========================================================
X_raw = data[must_cols].copy()

preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), must_cols)],
    remainder="drop"
)

# For demo output, show encoded feature names + first few rows
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = enc.fit_transform(X_raw)
X_encoded_df = pd.DataFrame(X_encoded, columns=enc.get_feature_names_out(must_cols))

print("\n==================== STEP 6 INPUT: X_raw (must context) ====================")
print(X_raw)
print("\n==================== STEP 6 OUTPUT: X_encoded (one-hot) ====================")
print(X_encoded_df)


# =========================================================
# STEP 7) Supervised training (CORE)
# Train one model per optional config:
# - binary: calibrated logistic regression -> P(y=1 | context)
# - multiclass: multinomial logistic regression -> P(class | context)
# =========================================================
# Binary: pilot
y_pilot = ds_pilot["y_pilot"].astype(int)
pilot_base = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=2000))])
pilot_model = CalibratedClassifierCV(pilot_base, method="sigmoid", cv=2)  # tiny demo CV
pilot_model.fit(X_raw, y_pilot)

# Binary: fridge
y_fridge = ds_fridge["y_fridge"].astype(int)
fridge_base = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=2000))])
fridge_model = CalibratedClassifierCV(fridge_base, method="sigmoid", cv=2)
fridge_model.fit(X_raw, y_fridge)

# Multiclass: audio
y_audio = ds_audio["y_audio"].astype(str)
audio_model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))
])
audio_model.fit(X_raw, y_audio)

# Multiclass: calipers
y_calipers = ds_calipers["y_calipers"].astype(str)
calipers_model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))
])
calipers_model.fit(X_raw, y_calipers)

print("\n==================== STEP 7 OUTPUT: trained model classes ====================")
print("pilot classes:", pilot_model.classes_)
print("fridge classes:", fridge_model.classes_)
print("audio classes:", audio_model.named_steps["clf"].classes_)
print("calipers classes:", calipers_model.named_steps["clf"].classes_)


# =========================================================
# STEP 8) Collaborative Filtering (item-based) using (config=value) tokens
# Purpose: bundle/co-purchase hints. Not personalized by must-context.
# =========================================================
# Build interactions from observed final orders:
# - multiclass always included as item token "config=value"
# - binary included only when 1 => "config=yes"
def order_to_items(r):
    items = []
    items.append(f"audio={r['audio']}")
    items.append(f"calipers={r['calipers']}")
    if int(r["pilot"]) == 1:
        items.append("pilot=yes")
    if int(r["fridge"]) == 1:
        items.append("fridge=yes")
    return items

interactions = []
for _, r in data.iterrows():
    for it in order_to_items(r):
        interactions.append({"order_id": r["order_id"], "user_id": r["user_id"], "item": it})
interactions = pd.DataFrame(interactions)

# user-item matrix
users = interactions["user_id"].unique().tolist()
items = sorted(interactions["item"].unique().tolist())
M = pd.DataFrame(0, index=users, columns=items)
for _, r in interactions.iterrows():
    M.loc[r["user_id"], r["item"]] = 1

# item-item cosine similarity
S = cosine_similarity(M.T) if len(items) > 1 else np.array([[1.0]])
S_df = pd.DataFrame(S, index=items, columns=items)

# top-2 similar items for each item
topk_rows = []
for it in items:
    sims = S_df[it].drop(index=it).sort_values(ascending=False).head(2)
    for sim_item, sim_val in sims.items():
        topk_rows.append({"item": it, "similar_item": sim_item, "cosine_sim": float(sim_val)})
topk = pd.DataFrame(topk_rows).sort_values(["item","cosine_sim"], ascending=[True, False])

print("\n==================== STEP 8 INPUT: interactions ====================")
print(interactions)
print("\n==================== STEP 8 OUTPUT A: user-item matrix M ====================")
print(M)
print("\n==================== STEP 8 OUTPUT B: item-item similarity (rounded) ====================")
print(S_df.round(3))
print("\n==================== STEP 8 OUTPUT C: top similar items ====================")
print(topk)


# =========================================================
# STEP 9) Scoring a new user (personalized supervised probabilities)
# + Produce salesperson report sections (Recommended / Optional / Do Not Recommend)
# =========================================================
new_user = wishlist[wishlist["user_id"] == "U200"].copy()
X_new = new_user[must_cols]

# Predict binary probabilities
p_pilot_yes = float(pilot_model.predict_proba(X_new)[0, 1])
p_fridge_yes = float(fridge_model.predict_proba(X_new)[0, 1])

# Predict multiclass distributions
audio_classes = audio_model.named_steps["clf"].classes_
audio_proba = audio_model.predict_proba(X_new)[0]
audio_dist = dict(zip(audio_classes, [float(p) for p in audio_proba]))

cal_classes = calipers_model.named_steps["clf"].classes_
cal_proba = calipers_model.predict_proba(X_new)[0]
cal_dist = dict(zip(cal_classes, [float(p) for p in cal_proba]))

# Pick best value for multiclass configs
audio_best = max(audio_dist.items(), key=lambda x: x[1])
cal_best = max(cal_dist.items(), key=lambda x: x[1])

print("\n==================== STEP 9 OUTPUT: supervised predictions for U200 ====================")
print("P(pilot=yes)  =", round(p_pilot_yes, 4))
print("P(fridge=yes) =", round(p_fridge_yes, 4))
print("P(audio=value)    =", {k: round(v,4) for k,v in audio_dist.items()})
print("P(calipers=value) =", {k: round(v,4) for k,v in cal_dist.items()})

# Simple policy to create sections (demo thresholds)
# In prod: tune thresholds by business and calibration
def bucket(p):
    if p >= 0.70: return "Recommended"
    if p >= 0.35: return "Optional"
    return "Do not recommend"

report_rows = []
report_rows.append({"section": bucket(p_pilot_yes), "config": "Pilot", "suggested_value": "Yes", "likelihood": p_pilot_yes})
report_rows.append({"section": bucket(p_fridge_yes), "config": "Fridge", "suggested_value": "Yes", "likelihood": p_fridge_yes})
report_rows.append({"section": bucket(audio_best[1]), "config": "Audio", "suggested_value": audio_best[0], "likelihood": audio_best[1]})
report_rows.append({"section": bucket(cal_best[1]), "config": "Calipers", "suggested_value": cal_best[0], "likelihood": cal_best[1]})

report = pd.DataFrame(report_rows).sort_values(["section","likelihood"], ascending=[True, False])

print("\n==================== STEP 9 OUTPUT: report rows (bucketed) ====================")
print(report)

# Add CF bundle hints for the suggested values (optional internal)
# Example: take the "item token" for recommended audio value and show nearest neighbors
audio_item = f"audio={audio_best[0]}"
bundle_hints = topk[topk["item"] == audio_item].copy()

print("\n==================== STEP 9 OUTPUT: CF bundle hints for best audio item ====================")
print(bundle_hints)


# =========================================================
# STEP 10) Final formatted salesperson report (text)
# =========================================================
customer_line = f"{new_user.iloc[0]['customer_name']} ({new_user.iloc[0]['user_id']})"
context_line = (f"{new_user.iloc[0]['car_type']} · {new_user.iloc[0]['battery']} · {new_user.iloc[0]['color']} · "
                f'{new_user.iloc[0]["wheel"]}" · {new_user.iloc[0]["interior_material"]} · '
                f'Nomi {new_user.iloc[0]["nomi"]} · Key {new_user.iloc[0]["key1_color"]}')

def fmt_section(sec):
    df = report[report["section"] == sec].copy()
    if df.empty:
        return f"{sec}: (none)\n"
    lines = [f"{sec}:"]
    for _, r in df.sort_values("likelihood", ascending=False).iterrows():
        lines.append(f"- {r['config']}: {r['suggested_value']}  (likelihood {r['likelihood']:.2f})")
    return "\n".join(lines) + "\n"

print("\n==================== STEP 10 OUTPUT: Salesperson Report ====================")
print(f"Customer: {customer_line}")
print(f"Context:  {context_line}\n")
print(fmt_section("Recommended"))
print(fmt_section("Optional"))
print(fmt_section("Do not recommend"))
print("Notes: likelihood = probability the option/value appears in the final order for similar contexts (no offer logs).")
