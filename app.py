
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ── Load artifacts ─────────────────────────────────────────────────────────
ARTIFACTS_PATH = "model_artifacts/hybrid_artifacts.pkl"
CSV_PATH       = "model_artifacts/courses.csv"

@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_PATH, "rb") as f:
        art = pickle.load(f)
    df = pd.read_csv(CSV_PATH)
    return art, df

art, df = load_artifacts()

cfm   = art["course_feature_matrix"]
cni   = art["course_names_index"]
uf    = art["user_factors"]
cf    = art["course_factors"]
pf    = art["pivot_filled"]
uids  = art["user_ids"]
ccols = art["course_cols"]

# ── Helper functions ────────────────────────────────────────────────────────
def content_recommend(course_name, top_n=5):
    idx      = cni.index(course_name)
    sims     = cosine_similarity(cfm[idx].reshape(1,-1), cfm)[0]
    series   = pd.Series(sims, index=cni).drop(index=course_name)
    return series.sort_values(ascending=False).head(top_n)

def collab_recommend(user_id, top_n=5):
    u_idx    = uids.index(user_id)
    preds    = uf[u_idx] @ cf.T
    scores   = pd.Series(preds, index=ccols)
    seen     = pf.loc[user_id]
    scores   = scores[seen == 0]
    return scores.sort_values(ascending=False).head(top_n)

def hybrid_recommend(user_id, liked_course, top_n=5, alpha=0.5):
    cb  = content_recommend(liked_course, top_n=len(cni)-1)
    u_idx   = uids.index(user_id)
    preds   = uf[u_idx] @ cf.T
    cf_ser  = pd.Series(preds, index=ccols)
    mm = MinMaxScaler()
    cb_n = pd.Series(mm.fit_transform(cb.values.reshape(-1,1)).flatten(), index=cb.index)
    cf_n = pd.Series(mm.fit_transform(cf_ser.values.reshape(-1,1)).flatten(), index=cf_ser.index)
    combined = alpha * cb_n.reindex(cf_n.index, fill_value=0) + (1-alpha) * cf_n
    seen_courses = pf.loc[user_id][pf.loc[user_id] > 0].index
    combined     = combined.drop(index=seen_courses, errors="ignore").drop(index=liked_course, errors="ignore")
    top = combined.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"Course": top.index, "Hybrid Score": top.values.round(4)})

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")
st.title("🎓 Online Course Recommendation System")
st.caption("Hybrid: Content-Based + Collaborative Filtering (SVD)")

st.sidebar.header("Settings")
mode    = st.sidebar.radio("Mode", ["Hybrid", "Content-Based Only", "Collaborative Only"])
top_n   = st.sidebar.slider("Number of Recommendations", 3, 15, 5)
alpha   = st.sidebar.slider("Hybrid Weight α (Content ↔ Collab)", 0.0, 1.0, 0.5, 0.1,
                             help="1.0 = pure content-based, 0.0 = pure collaborative")

col1, col2 = st.columns(2)

with col1:
    user_id = st.selectbox("Select User ID", uids)

with col2:
    liked_course = st.selectbox("Select a Course You Liked", sorted(cni))

if st.button("Get Recommendations", type="primary"):
    st.divider()

    if mode == "Hybrid":
        st.subheader(f"🔀 Hybrid Recommendations (α={alpha})")
        recs = hybrid_recommend(user_id, liked_course, top_n=top_n, alpha=alpha)
        st.dataframe(recs, use_container_width=True)

        # Show course metadata
        st.subheader("📋 Course Details")
        details = df[df["course_name"].isin(recs["Course"])][[
            "course_name", "difficulty_level", "course_duration_hours",
            "course_price", "rating", "certification_offered"
        ]].drop_duplicates("course_name")
        st.dataframe(details.rename(columns={
            "course_name": "Course", "difficulty_level": "Difficulty",
            "course_duration_hours": "Duration (hrs)", "course_price": "Price ($)",
            "rating": "Avg Rating", "certification_offered": "Certificate"
        }), use_container_width=True)

    elif mode == "Content-Based Only":
        st.subheader("📄 Content-Based Recommendations")
        cb = content_recommend(liked_course, top_n=top_n)
        st.dataframe(pd.DataFrame({"Course": cb.index, "Similarity": cb.values.round(4)}),
                     use_container_width=True)

    else:
        st.subheader("👥 Collaborative Recommendations")
        cf_r = collab_recommend(user_id, top_n=top_n)
        st.dataframe(pd.DataFrame({"Course": cf_r.index, "Predicted Score": cf_r.values.round(4)}),
                     use_container_width=True)

# ── User history sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📚 User History")
    history = df[df["user_id"] == user_id][["course_name", "rating", "difficulty_level"]].drop_duplicates()
    st.dataframe(history.rename(columns={"course_name": "Course", "rating": "Rating",
                                          "difficulty_level": "Level"}),
                 use_container_width=True, height=300)
