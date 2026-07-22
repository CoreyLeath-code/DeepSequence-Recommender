"""DeepSequence Recommender – Streamlit demo surface.

Run locally:
    streamlit run streamlit_app.py

The app initialises the same model used by the FastAPI service and lets you
interactively build an item-sequence and receive top-k recommendations.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from app.core.artifacts import load_bundle
from app.core.config import settings
from app.core.data_processor import SequenceProcessor
from app.core.model import DeepSequenceModel

DEMO_CATALOGUE_SIZE = 200


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DeepSequence Recommender",
    page_icon="🎯",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Model initialisation (cached so it only runs once per session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading recommendation model…")
def load_model() -> tuple[SequenceProcessor, DeepSequenceModel, str, bool]:
    """Load a trained bundle or an explicitly labelled development model."""
    bundle = Path(settings.model_bundle_path)
    if (bundle / "manifest.json").is_file():
        processor, model, manifest = load_bundle(bundle)
        return processor, model, manifest.model_version, True
    torch.manual_seed(7)
    processor = SequenceProcessor(max_length=settings.max_sequence_length)
    demo_items = [[f"item_{i}" for i in range(DEMO_CATALOGUE_SIZE)]]
    processor.fit(demo_items)

    model = DeepSequenceModel(
        num_items=processor.vocab_size,
        embedding_dim=settings.embedding_dim,
        hidden_dim=settings.hidden_dim,
        num_layers=settings.num_layers,
    )
    model.eval()
    return processor, model, "development-untrained", False


processor, model, model_version, trained = load_model()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎯 DeepSequence Recommender")
st.markdown(
    "A **sequence-aware** deep learning recommender powered by a "
    "Bidirectional LSTM + Attention architecture."
)
if not trained:
    st.warning(
        "Development mode: this seeded model is untrained and does not represent ranking quality."
    )
st.divider()

# Sidebar: model info
with st.sidebar:
    st.header("Model info")
    st.metric("Vocab size", processor.vocab_size)
    st.metric("Embedding dim", settings.embedding_dim)
    st.metric("Hidden dim", settings.hidden_dim)
    st.metric("LSTM layers", settings.num_layers)
    st.metric("Max sequence length", settings.max_sequence_length)
    st.caption(f"Model version: {model_version}")
    st.divider()
    st.caption("Architecture: BiLSTM + scaled dot-product attention → top-k items")

# Main pane
st.subheader("Build your interaction sequence")

available_items = [f"item_{i}" for i in range(DEMO_CATALOGUE_SIZE)]
selected_items: list[str] = st.multiselect(
    "Select items you have interacted with (in order):",
    options=available_items,
    default=["item_0", "item_5", "item_12"],
    help="These represent your interaction history. The model predicts what comes next.",
)

top_k = st.slider("Number of recommendations (top-k)", min_value=1, max_value=20, value=10)

st.divider()

if st.button("🚀 Get Recommendations", use_container_width=True):
    if not selected_items:
        st.warning("Please select at least one item to build a sequence.")
    else:
        with st.spinner("Running inference…"):
            tensor = processor.to_tensor(selected_items)
            exclude_ids = [processor.item_to_idx(i) for i in selected_items]
            raw_indices = model.recommend(tensor, top_k=top_k, exclude_ids=exclude_ids)
            recommendations = processor.decode_recommendations(raw_indices)
            filtered = [r for r in recommendations if r is not None]

        st.success(f"Top-{top_k} recommendations for your sequence:")
        for rank, item in enumerate(filtered, start=1):
            st.write(f"**{rank}.** {item}")

        st.divider()
        st.caption(
            f"Input sequence length: {len(selected_items)} | "
            f"Recommendations returned: {len(filtered)}"
        )

st.divider()
st.markdown(
    "**How it works:** The model encodes your item history through an embedding layer, "
    "processes it with a Bidirectional LSTM, applies scaled dot-product attention to "
    "weight the most relevant interactions, and projects the result onto the item "
    "vocabulary to produce ranked scores. Items already in your history are excluded."
)
