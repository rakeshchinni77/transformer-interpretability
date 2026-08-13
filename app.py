import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config.settings import settings
from model import TransformerClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Transformer Interpretability Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and dark glassmorphism layout
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads and caches the trained custom TransformerClassifier model and BERT tokenizer.
    Only the tokenizer is loaded from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name)

    model_path = Path(settings.model_path)
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent / "models" / "final_model.pth"

    device = torch.device(settings.resolved_device)

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=settings.d_model,
        num_heads=settings.num_heads,
        num_layers=settings.num_layers,
        d_ff=settings.d_ff,
        dropout=settings.dropout,
        num_classes=settings.num_classes,
        max_len=settings.max_len,
    )

    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    st.title("⚡ Transformer Encoder Interpretability Dashboard")
    st.markdown(
        "Explore internal multi-head self-attention weights and head entropy metrics from our "
        "PyTorch Transformer Encoder built **from scratch** and trained on the Stanford IMDB dataset."
    )

    # Load cached model & tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        st.stop()

    # Sidebar Controls
    st.sidebar.header("🕹️ Attention Controls")
    num_layers = model.encoder.layers.__len__()
    num_heads = model.encoder.layers[0].self_attn.h

    layer_idx = st.sidebar.selectbox(
        "Select Layer",
        options=list(range(num_layers)),
        format_func=lambda x: f"Layer {x}",
        index=0,
    )

    head_idx = st.sidebar.selectbox(
        "Select Attention Head",
        options=list(range(num_heads)),
        format_func=lambda x: f"Head {x}",
        index=0,
    )

    # Main Form Input
    with st.form(key="review_form"):
        st.subheader("📝 Input Review for Sentiment & Attention Analysis")
        review_text = st.text_area(
            "Enter a sentence/review",
            value="This movie was surprisingly good and the acting was excellent.",
            height=100,
        )
        submit_button = st.form_submit_button(label="Analyze Review")

    # Run inference
    encoded = tokenizer(
        review_text,
        padding="max_length",
        truncation=True,
        max_length=settings.max_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits, attentions = model(input_ids, attention_mask=attention_mask)

    probs = F.softmax(logits, dim=-1)[0]
    pred_class = torch.argmax(logits, dim=-1).item()
    confidence = probs[pred_class].item() * 100.0
    sentiment = "Positive" if pred_class == 1 else "Negative"

    # Display Prediction Banner
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Sentiment", sentiment)
    with col2:
        st.metric("Confidence Score", f"{confidence:.2f}%")
    with col3:
        st.metric("Negative / Positive Probabilities", f"{probs[0]:.2f} / {probs[1]:.2f}")

    st.markdown("---")

    # Extract non-padded tokens and attention matrix
    valid_len = int(attention_mask[0].sum().item())
    valid_ids = input_ids[0][:valid_len].tolist()
    tokens = tokenizer.convert_ids_to_tokens(valid_ids)

    attn_matrix = attentions[layer_idx][0, head_idx, :valid_len, :valid_len].cpu().numpy()

    # Heatmap Section with MANDATORY DOM Test ID
    st.subheader(f"🔥 Attention Heatmap (Layer {layer_idx} — Head {head_idx})")
    st.markdown(
        '<div data-testid="attention-heatmap-container" id="attention-heatmap-container">',
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(max(6, valid_len * 0.6), max(5, valid_len * 0.5)))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=(valid_len <= 15),
        fmt=".2f" if valid_len <= 15 else "",
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_xlabel("Key / Attended Token", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("Query Token", color="#e0e0e0", fontsize=12)
    ax.set_title(f"Layer {layer_idx} — Head {head_idx} Self-Attention", color="#ffffff", fontsize=14)
    ax.tick_params(colors="#e0e0e0", labelsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Entropy Dashboard Section with MANDATORY DOM Test ID
    st.subheader("📊 Attention Entropy Dashboard")
    st.markdown(
        '<div data-testid="entropy-dashboard-container" id="entropy-dashboard-container">',
        unsafe_allow_html=True,
    )

    csv_path = Path(__file__).resolve().parent / "logs" / "training_metrics.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            epochs = sorted(df["epoch"].unique())

            selected_epoch = st.select_slider(
                "Select Epoch to view Head Entropies",
                options=epochs,
                value=epochs[-1],
            )

            df_epoch = df[df["epoch"] == selected_epoch].copy()
            df_epoch["layer_head"] = df_epoch.apply(
                lambda r: f"L{int(r['layer'])} H{int(r['head'])}", axis=1
            )

            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig_entropy, ax_e = plt.subplots(figsize=(8, 4))
                fig_entropy.patch.set_facecolor('#0e1117')
                ax_e.set_facecolor('#0e1117')

                bars = ax_e.bar(
                    df_epoch["layer_head"],
                    df_epoch["attention_entropy"],
                    color="#6366f1",
                    edgecolor="#4f46e5",
                )
                ax_e.set_ylabel("Shannon Entropy H(p)", color="#e0e0e0", fontsize=11)
                ax_e.set_xlabel("Layer / Head Combination", color="#e0e0e0", fontsize=11)
                ax_e.set_title(
                    f"Head Entropy Distribution (Epoch {selected_epoch})",
                    color="#ffffff",
                    fontsize=12,
                )
                ax_e.tick_params(colors="#e0e0e0")
                ax_e.set_ylim(0, math.log(settings.max_len) * 1.1)

                for bar in bars:
                    yval = bar.get_height()
                    ax_e.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        yval + 0.05,
                        f"{yval:.2f}",
                        ha="center",
                        va="bottom",
                        color="#ffffff",
                        fontsize=9,
                    )

                plt.tight_layout()
                st.pyplot(fig_entropy)
                plt.close(fig_entropy)

            with col_table:
                st.write(f"**Metrics Summary (Epoch {selected_epoch})**")
                st.dataframe(
                    df_epoch[["layer", "head", "attention_entropy"]].reset_index(drop=True),
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(f"Could not load training metrics CSV: {e}")
    else:
        st.info("No logs/training_metrics.csv found. Run training to populate entropy metrics.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
