"""
app.py
======
Day 6 — Analyst dashboard (Streamlit).

Reads Day 3/4/5 output (scored_events.csv, day4_results.csv, drift_log.json,
entities.json) and renders:

  1. Live Alert Feed          - filterable, sorted by risk, color-coded
  2. Alert Detail             - SHAP top-factors + plain-language Signal
                                 Breakdown (time / geo / device / graph / velocity)
  3. Geo + Graph view         - impossible_travel on a world map,
                                 lateral_movement on the access-topology graph
  4. Entity Timeline          - one user's risk score over the full simulation
  5. Model Health             - drift status, cold-start indicator, live
                                 classifier metrics, retrain log

Sidebar has a "Retrain pipeline live" button that actually re-runs
run_day3.main() + run_day4.main() (real training, not a simulated log line -
same as drift.py's trigger_retrain()) and streams the console output into
an expander so the retrain is visibly real, not decorative.

Run from the `src/` directory:
    streamlit run app.py
"""

import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # so sibling modules (run_day3, explain, ...) import regardless of cwd

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

st.set_page_config(page_title="Behavioral Anomaly Detection", layout="wide", page_icon="🛡️")

# ---------------------------------------------------------------------------
# Signal-category grouping (for the plain-language "Signal Breakdown" panel).
# Maps the 13 raw behavioral features onto the 5 human-facing signal
# families the pitch talks about: time / geo / device / graph / velocity.
# ---------------------------------------------------------------------------
SIGNAL_GROUPS = {
    "time":     ["hour_zscore"],
    "geo":      ["geo_distance_km", "geo_velocity_kmh"],
    "device":   ["is_new_device", "fingerprint_mismatch"],
    "graph":    ["is_new_host", "is_foreign_dept", "is_first_time_in_this_dept",
                 "foreign_dept_burst_distinct", "new_host_burst_count", "graph_dist_from_history"],
    "velocity": ["time_since_last_event_min", "failure_burst_count"],
}
SIGNAL_ICON = {"time": "🕐", "geo": "🌍", "device": "💻", "graph": "🕸️", "velocity": "⚡"}
FEATURE_TO_GROUP = {f: g for g, feats in SIGNAL_GROUPS.items() for f in feats}

DEFENSE_IN_DEPTH_NOTE = (
    "Behavioral detection increases attacker cost — evading it requires simultaneously "
    "faking multiple independent signals (time, geo, device, graph, velocity). No single-layer "
    "system is unbeatable; this is intended as one layer in defense-in-depth, not a silver bullet."
)


# ---------------------------------------------------------------------------
# Data loading (cached — cleared after a live retrain)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_results():
    path = os.path.join(DATA_DIR, "day4_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False, parse_dates=["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_geo_joined():
    """day4_results.csv has no lat/lon — join back to synthetic_logs.csv on
    event_id to get geo_lat/geo_lon/geo_city/geo_country for the map view."""
    logs_path = os.path.join(DATA_DIR, "synthetic_logs.csv")
    res = load_results()
    if res is None or not os.path.exists(logs_path):
        return None
    logs = pd.read_csv(logs_path, usecols=["event_id", "geo_lat", "geo_lon", "geo_city", "geo_country", "source_ip"])
    return res.merge(logs, on="event_id", how="left")


@st.cache_data(show_spinner=False)
def load_entities():
    path = os.path.join(DATA_DIR, "entities.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_drift_log():
    path = os.path.join(DATA_DIR, "drift_log.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_explainer():
    """Cached separately from data (cache_resource, not cache_data) since it
    holds a live model object, not serializable data."""
    from explain import AlertExplainer
    return AlertExplainer()


def artifacts_exist():
    return os.path.exists(os.path.join(MODELS_DIR, "xgb_classifier.joblib")) and \
        os.path.exists(os.path.join(DATA_DIR, "day4_results.csv"))


# ---------------------------------------------------------------------------
# Signal breakdown (SHAP-backed when the model is available, feature-flag
# fallback otherwise so the panel never just breaks)
# ---------------------------------------------------------------------------
def signal_breakdown_for_row(row, explainer=None):
    """Returns {group: pct_contribution} summing to ~100, plus the raw
    per-feature SHAP sentences when an explainer is available."""
    feature_cols = list(FEATURE_TO_GROUP.keys())
    sentences = []

    if explainer is not None:
        try:
            feat_row = row[explainer.feature_cols].astype(float).fillna(0).to_numpy(dtype=float)
            pred = row.get("predicted_attack_type") or "normal"
            if pred not in explainer.classes:
                pred = "normal"
            result = explainer.explain_row(feat_row, pred, top_k=len(explainer.feature_cols))
            group_totals = {g: 0.0 for g in SIGNAL_GROUPS}
            for f in result["top_factors"]:
                g = FEATURE_TO_GROUP.get(f["feature"])
                if g:
                    group_totals[g] += abs(f["shap_value"])
            total = sum(group_totals.values()) or 1e-9
            pct = {g: round(100 * v / total, 1) for g, v in group_totals.items()}
            sentences = sorted(result["top_factors"], key=lambda x: -abs(x["shap_value"]))[:3]
            return pct, sentences
        except Exception:
            pass  # fall through to the flag-based fallback below

    # Fallback: no trained explainer available yet - use simple normalized
    # flag/z-score magnitudes as a stand-in so the panel still renders.
    raw = {
        "hour_zscore": min(abs(row.get("hour_zscore", 0)), 5) / 5,
        "geo_distance_km": min(row.get("geo_distance_km", 0), 5000) / 5000,
        "geo_velocity_kmh": min(row.get("geo_velocity_kmh", 0), 2000) / 2000,
        "is_new_device": row.get("is_new_device", 0),
        "fingerprint_mismatch": row.get("fingerprint_mismatch", 0),
        "is_new_host": row.get("is_new_host", 0),
        "is_foreign_dept": row.get("is_foreign_dept", 0),
        "is_first_time_in_this_dept": row.get("is_first_time_in_this_dept", 0),
        "foreign_dept_burst_distinct": min(row.get("foreign_dept_burst_distinct", 0), 4) / 4,
        "new_host_burst_count": min(row.get("new_host_burst_count", 0), 5) / 5,
        "graph_dist_from_history": min(row.get("graph_dist_from_history", 0), 3) / 3,
        "time_since_last_event_min": 1 - min(row.get("time_since_last_event_min", 60), 60) / 60,
        "failure_burst_count": min(row.get("failure_burst_count", 0), 5) / 5,
    }
    group_totals = {g: 0.0 for g in SIGNAL_GROUPS}
    for f, v in raw.items():
        g = FEATURE_TO_GROUP.get(f)
        if g:
            group_totals[g] += float(v)
    total = sum(group_totals.values()) or 1e-9
    pct = {g: round(100 * v / total, 1) for g, v in group_totals.items()}
    return pct, sentences


# ---------------------------------------------------------------------------
# Sidebar - pipeline controls + at-a-glance health
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🛡️ Threat Ops Console")

    if not artifacts_exist():
        st.sidebar.warning("No trained model artifacts found yet. Run the pipeline once.")

    if st.sidebar.button("🔄 Retrain pipeline live (Day 3 + Day 4)", use_container_width=True):
        log_buf = io.StringIO()
        with st.sidebar:
            with st.spinner("Retraining Isolation Forest + Autoencoder + XGBoost on the full dataset..."):
                t0 = time.time()
                try:
                    with contextlib.redirect_stdout(log_buf):
                        import run_day3, run_day4
                        import importlib
                        importlib.reload(run_day3)
                        importlib.reload(run_day4)
                        run_day3.main()
                        run_day4.main()
                    elapsed = time.time() - t0
                    st.success(f"Retrained end-to-end in {elapsed:.1f}s")
                except Exception as e:
                    st.error(f"Retrain failed: {e}")
        st.session_state["last_retrain_log"] = log_buf.getvalue()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if "last_retrain_log" in st.session_state:
        with st.sidebar.expander("Last retrain log"):
            st.text(st.session_state["last_retrain_log"][-4000:])

    st.sidebar.divider()

    drift_log = load_drift_log()
    entities = load_entities()
    if drift_log:
        n_fired = len(drift_log.get("per_entity_first_fire_day", {}))
        alerts = drift_log.get("population_alerts", [])
        status = "🔴 DRIFT ALERT" if alerts else "🟢 Stable"
        st.sidebar.metric("Drift status", status, delta=f"{n_fired} entities drifted")
    if entities:
        n_cold = sum(1 for v in entities["users"].values() if v.get("is_cold_start"))
        st.sidebar.metric("Cold-start entities", n_cold)

    st.sidebar.divider()
    st.sidebar.caption(DEFENSE_IN_DEPTH_NOTE)


# ---------------------------------------------------------------------------
# Tab 1 - Live Alert Feed
# ---------------------------------------------------------------------------
def render_alert_feed(df):
    st.subheader("🚨 Live Alert Feed")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_risk = st.slider("Min unified risk score", 0, 100, 50)
    with c2:
        types = ["(any)"] + sorted(df["predicted_attack_type"].dropna().unique().tolist())
        type_filter = st.selectbox("Predicted attack type", types)
    with c3:
        split_filter = st.selectbox("Split", ["(any)", "test", "val", "train"])
    with c4:
        review_only = st.checkbox("Secondary review only")

    view = df[df["unified_risk_score"] >= min_risk].copy()
    if type_filter != "(any)":
        view = view[view["predicted_attack_type"] == type_filter]
    if split_filter != "(any)":
        view = view[view["clf_split"] == split_filter]
    if review_only:
        view = view[view["secondary_review_flag"] == True]  # noqa: E712

    view = view.sort_values("unified_risk_score", ascending=False)
    st.caption(f"{len(view):,} alerts match current filters (of {len(df):,} total events)")

    show_cols = ["timestamp", "user_id", "dept", "host", "predicted_attack_type",
                 "unified_risk_score", "classifier_confidence", "final_risk_score",
                 "secondary_review_flag", "label", "attack_type", "event_id"]
    st.dataframe(
        view[show_cols].head(300).style.background_gradient(subset=["unified_risk_score"], cmap="Reds"),
        use_container_width=True, height=420,
    )
    return view


# ---------------------------------------------------------------------------
# Tab 2 - Alert Detail / Explainability
# ---------------------------------------------------------------------------
def render_alert_detail(df):
    st.subheader("🔍 Alert Detail & Explainability")

    candidates = df[df["predicted_attack_type"].notna()].sort_values("unified_risk_score", ascending=False)
    if candidates.empty:
        st.info("No attack-predicted alerts to explain yet.")
        return

    options = candidates["event_id"].head(200).tolist()
    labels = {
        eid: f"{row.user_id} · {row.predicted_attack_type} · risk={row.unified_risk_score:.1f}"
        for eid, row in zip(options, candidates.head(200).itertuples())
    }
    selected = st.selectbox("Select an alert", options, format_func=lambda x: labels[x])
    row = df[df["event_id"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unified risk score", f"{row['unified_risk_score']:.1f}")
    c2.metric("Predicted type", str(row["predicted_attack_type"]))
    c3.metric("Classifier confidence", f"{row['classifier_confidence']*100:.1f}%")
    c4.metric("True label (ground truth)", str(row["attack_type"]) if pd.notna(row["attack_type"]) else "normal")

    explainer = None
    if artifacts_exist():
        try:
            explainer = load_explainer()
        except Exception as e:
            st.warning(f"SHAP explainer unavailable: {e}")

    pct, sentences = signal_breakdown_for_row(row, explainer)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Signal Breakdown** — which independent channels fired")
        fig = go.Figure(go.Bar(
            x=list(pct.values()), y=[f"{SIGNAL_ICON[g]} {g}" for g in pct.keys()],
            orientation="h", marker_color="#c0392b",
        ))
        fig.update_layout(xaxis_title="% contribution to risk", height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(DEFENSE_IN_DEPTH_NOTE)

    with right:
        st.markdown("**Top contributing factors (SHAP, exact for XGBoost)**")
        if sentences:
            for s in sentences:
                direction = "🔺" if s["shap_value"] > 0 else "🔻"
                st.write(f"{direction} {s['sentence']}")
        else:
            st.info("Retrain the pipeline once to enable exact SHAP attributions for this alert.")

    with st.expander("Raw feature values for this event"):
        feat_cols = list(FEATURE_TO_GROUP.keys())
        st.dataframe(row[feat_cols].to_frame("value"), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 - Geo (impossible_travel) + Graph (lateral_movement)
# ---------------------------------------------------------------------------
def render_geo_graph(df, entities):
    st.subheader("🗺️ Geo & 🕸️ Graph views")
    geo_df = load_geo_joined()

    left, right = st.columns(2)

    with left:
        st.markdown("**Impossible travel — flagged logins on the map**")
        if geo_df is not None:
            sub = geo_df[geo_df["predicted_attack_type"] == "impossible_travel"]
            if sub.empty:
                sub = geo_df[geo_df["attack_type"] == "impossible_travel"]
            if not sub.empty:
                fig = px.scatter_geo(
                    sub, lat="geo_lat", lon="geo_lon", color="unified_risk_score",
                    hover_name="user_id", hover_data=["geo_city", "geo_country", "timestamp"],
                    color_continuous_scale="Reds", projection="natural earth",
                )
                fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No impossible_travel alerts in the current data.")
        else:
            st.info("synthetic_logs.csv not found — can't join geo coordinates.")

    with right:
        st.markdown("**Lateral movement — access-topology graph**")
        if entities is None:
            st.info("entities.json not found.")
        else:
            lm_events = df[(df["predicted_attack_type"] == "lateral_movement") |
                            (df["attack_type"] == "lateral_movement")]
            users_with_lm = lm_events["user_id"].dropna().unique().tolist()
            if not users_with_lm:
                st.info("No lateral_movement alerts to visualize.")
            else:
                picked_user = st.selectbox("User", sorted(users_with_lm))
                fig = build_graph_figure(entities, df, picked_user)
                st.plotly_chart(fig, use_container_width=True)


def build_graph_figure(entities, df, user_id):
    import networkx as nx
    G = nx.Graph()
    for u, v, attrs in entities["graph_edges"]:
        G.add_edge(u, v, **attrs)

    normal_hosts = set(entities["users"].get(user_id, {}).get("normal_hosts", []))
    touched = df[df["user_id"] == user_id]["host"].dropna().unique().tolist()
    touched_set = set(touched)

    # keep the visualization readable: normal hosts + touched hosts + immediate neighbors
    keep = set(normal_hosts) | touched_set
    for h in list(touched_set):
        if h in G:
            keep |= set(G.neighbors(h))
    sub = G.subgraph(keep)

    pos = nx.spring_layout(sub, seed=42, k=0.6)
    edge_x, edge_y = [], []
    for a, b in sub.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#888"), hoverinfo="none")

    node_x, node_y, colors, texts = [], [], [], []
    for n in sub.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        if n in touched_set:
            colors.append("#c0392b")   # actually accessed during the incident (foreign/suspicious)
        elif n in normal_hosts:
            colors.append("#2980b9")   # normal baseline host for this user
        else:
            colors.append("#bdc3c7")   # context neighbor, not touched
        texts.append(n)
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=texts,
                             textposition="top center", hoverinfo="text",
                             marker=dict(size=14, color=colors, line=dict(width=1, color="white")))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=420, margin=dict(l=0, r=0, t=10, b=0),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ---------------------------------------------------------------------------
# Tab 4 - Entity timeline
# ---------------------------------------------------------------------------
def render_entity_timeline(df):
    st.subheader("📈 Entity Timeline")
    user_id = st.selectbox("Select entity", sorted(df["user_id"].dropna().unique()))
    sub = df[df["user_id"] == user_id].sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["unified_risk_score"], mode="lines+markers",
                              name="unified_risk_score", line=dict(color="#7f8c8d")))
    attacks = sub[sub["label"] == "attack"]
    if not attacks.empty:
        fig.add_trace(go.Scatter(x=attacks["timestamp"], y=attacks["unified_risk_score"], mode="markers",
                                  name="true attack event", marker=dict(color="#c0392b", size=11, symbol="x")))
    flagged = sub[sub["predicted_attack_type"].notna()]
    if not flagged.empty:
        fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["unified_risk_score"], mode="markers",
                                  name="model-flagged", marker=dict(color="#e67e22", size=9, symbol="circle-open")))
    fig.update_layout(height=420, yaxis_title="unified_risk_score", xaxis_title="time",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"{len(sub)} total events · {len(attacks)} true attack events · "
               f"{len(flagged)} model-flagged events for {user_id}")


# ---------------------------------------------------------------------------
# Tab 5 - Model Health
# ---------------------------------------------------------------------------
def render_model_health(df, drift_log, entities):
    st.subheader("⚙️ Model Health")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Drift detection**")
        if drift_log:
            val = drift_log["validation_against_ground_truth"]
            st.write(f"TP={val['tp']}  FP={val['fp']}  FN={val['fn']}  TN={val['tn']}")
            recall = val["tp"] / max(val["tp"] + val["fn"], 1)
            st.write(f"Recall: {recall*100:.1f}%  ·  False-alarm rate: {val['fp']/max(val['fp']+val['tn'],1)*100:.1f}%")
            if drift_log["population_alerts"]:
                alerts_df = pd.DataFrame(drift_log["population_alerts"])
                st.dataframe(alerts_df, use_container_width=True, height=160)
        else:
            st.info("drift_log.json not found — run drift.py.")

    with c2:
        st.markdown("**Cold-start entities**")
        if entities:
            cold = [u for u, v in entities["users"].items() if v.get("is_cold_start")]
            st.write(f"{len(cold)} entities currently cold-start: {', '.join(cold)}")
            cold_rows = df[df["is_cold_start_entity"] == 1]
            if len(cold_rows):
                st.write(f"Mean final_risk_score for cold-start normal rows: "
                         f"{cold_rows.loc[cold_rows['label']=='normal','final_risk_score'].mean():.1f}")
        else:
            st.info("entities.json not found.")

    with c3:
        st.markdown("**Secondary review tier**")
        n_flag = df["secondary_review_flag"].sum()
        n_flag_real = ((df["secondary_review_flag"]) & (df["label"] == "attack")).sum()
        st.write(f"{n_flag} rows flagged, {n_flag_real} real attacks recovered "
                 f"that the classifier alone missed.")

    st.divider()
    st.markdown("**Classifier performance (test split, recomputed live from current data)**")
    test = df[df["clf_split"] == "test"]
    if len(test):
        from sklearn.metrics import classification_report
        y_true = test["attack_type"].fillna("normal")
        y_pred = test["predicted_attack_type"].fillna("normal")
        report = classification_report(y_true, y_pred, digits=3, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("No test-split rows found — run the pipeline once.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    st.title("Behavioral Anomaly Detection — Analyst Console")

    df = load_results()
    if df is None:
        st.warning("No day4_results.csv found yet. Click **Retrain pipeline live** in the sidebar to generate it.")
        return

    entities = load_entities()
    drift_log = load_drift_log()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🚨 Alert Feed", "🔍 Alert Detail", "🗺️ Geo / Graph", "📈 Entity Timeline", "⚙️ Model Health"]
    )
    with tab1:
        render_alert_feed(df)
    with tab2:
        render_alert_detail(df)
    with tab3:
        render_geo_graph(df, entities)
    with tab4:
        render_entity_timeline(df)
    with tab5:
        render_model_health(df, drift_log, entities)


if __name__ == "__main__":
    main()
