import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import keras
import plotly.graph_objects as go


# ----------------------------------------------------------------------------
# THEME / COLOUR PALETTE
# ----------------------------------------------------------------------------
HONEY = "#F4A300"
HONEY_DARK = "#C9810A"
DEEP_BROWN = "#2B1D0E"


def get_theme_colors():
    """Resolve the active Streamlit theme into a dict of literal hex colors.

    We deliberately avoid relying purely on CSS custom-property inheritance
    for text-bearing elements (sidebar cards, result card, footer) because
    Streamlit renders different parts of the app into separate DOM subtrees,
    and custom properties don't always reach every subtree reliably. Instead
    we resolve the theme once in Python and bake literal colors directly
    into the HTML/CSS we emit, with !important to win any specificity fights.
    """
    is_dark = st.get_option("theme.base") == "dark"

    if is_dark:
        return {
            "is_dark": True,
            "accent": "#FFD166",
            "card_bg": "rgba(255,255,255,0.06)",
            "card_border": "rgba(244,163,0,0.30)",
            "sidebar_text": "#F0F0F0",
            "sidebar_heading": HONEY,
            "sidebar_link": HONEY,
            "result_bg_a": "rgba(244,163,0,0.20)",
            "result_bg_b": "rgba(244,163,0,0.05)",
            "result_genus": "#FFF8E7",
            "result_conf": "#E5E5E5",
            "footer_color": "#AAAAAA",
            "tab_hover_bg": "rgba(244,163,0,0.10)",
        }
    else:
        return {
            "is_dark": False,
            "accent": "#C9810A",
            "card_bg": "rgba(244,163,0,0.10)",
            "card_border": "rgba(200,120,0,0.35)",
            "sidebar_text": "#2B1D0E",
            "sidebar_heading": HONEY_DARK,
            "sidebar_link": HONEY_DARK,
            "result_bg_a": "rgba(244,163,0,0.16)",
            "result_bg_b": "rgba(244,163,0,0.04)",
            "result_genus": "#1A1000",
            "result_conf": "#3A2800",
            "footer_color": "#7A6040",
            "tab_hover_bg": "rgba(244,163,0,0.12)",
        }


def inject_custom_css(theme):
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Nunito+Sans:wght@400;600&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Nunito Sans', sans-serif;
            }}

            /* ================================================================
               HERO BANNER  — always dark-on-photo so text stays readable
               regardless of site theme
               ================================================================ */
            .hero-banner {{
                background: linear-gradient(
                    135deg,
                    #2B1D0E 0%,
                    #4a2f12 55%,
                    {HONEY_DARK} 100%
                );
                border-radius: 18px;
                padding: 2.5rem 2.5rem 2rem 2.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 6px 24px rgba(0,0,0,0.30);
                position: relative;
                overflow: hidden;
            }}
            .hero-banner::after {{
                content: "";
                position: absolute;
                right: -40px; top: -40px;
                width: 220px; height: 220px;
                background: radial-gradient(circle, rgba(244,163,0,0.30) 0%, rgba(244,163,0,0) 70%);
                border-radius: 50%;
            }}
            .hero-title {{
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
                font-size: 2.6rem;
                color: #FFF8E7 !important;
                margin-bottom: 0.25rem;
                letter-spacing: 0.5px;
            }}
            .hero-subtitle {{
                font-family: 'Poppins', sans-serif;
                font-weight: 600;
                font-size: 1.1rem;
                color: #FFD166 !important;
                margin-bottom: 0.5rem;
                letter-spacing: 0.5px;
            }}
            .hero-tagline {{
                color: rgba(255,248,231,0.88) !important;
                font-size: 0.97rem;
                max-width: 640px;
                line-height: 1.55;
            }}

            /* ================================================================
               SIDEBAR CARDS
               Fixed, self-contained LIGHT cream surface with dark text —
               stays the same warm cream box in both light and dark site
               themes, exactly like the reference design.
               ================================================================ */
            section[data-testid="stSidebar"] {{
                border-right: 1px solid rgba(244,163,0,0.30);
            }}
            .sidebar-card {{
                background: #F7F0E4;
                border: 1px solid rgba(200,120,0,0.22);
                border-radius: 12px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.9rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            }}
            section[data-testid="stSidebar"] .sidebar-card h5,
            .sidebar-card h5 {{
                font-family: 'Poppins', sans-serif;
                color: {HONEY_DARK} !important;
                -webkit-text-fill-color: {HONEY_DARK} !important;
                margin: 0 0 0.45rem 0;
                font-size: 0.9rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }}
            section[data-testid="stSidebar"] .sidebar-card p,
            section[data-testid="stSidebar"] .sidebar-card li,
            section[data-testid="stSidebar"] div.sidebar-card p,
            section[data-testid="stSidebar"] div.sidebar-card li,
            .sidebar-card p,
            .sidebar-card li,
            .sidebar-card ol,
            .sidebar-card ul {{
                font-size: 0.88rem !important;
                line-height: 1.5rem !important;
                color: #2B1D0E !important;
                -webkit-text-fill-color: #2B1D0E !important;
            }}
            section[data-testid="stSidebar"] .sidebar-card strong,
            .sidebar-card strong {{
                color: #1A1000 !important;
                -webkit-text-fill-color: #1A1000 !important;
            }}
            section[data-testid="stSidebar"] .sidebar-card a,
            .sidebar-card a {{
                color: {HONEY_DARK} !important;
                -webkit-text-fill-color: {HONEY_DARK} !important;
                text-decoration: none;
                font-weight: 600;
            }}
            .sidebar-card a:hover {{
                text-decoration: underline;
            }}

            /* ================================================================
               TABS
               ================================================================ */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            .stTabs [data-baseweb="tab"] {{
                background-color: {theme['tab_hover_bg']};
                border-radius: 10px 10px 0 0;
                padding: 10px 22px;
                font-weight: 600;
                font-size: 1rem;
            }}
            .stTabs [aria-selected="true"] {{
                background-color: rgba(244,163,0,0.18) !important;
                color: {theme['accent']} !important;
                border-bottom: 3px solid {HONEY} !important;
            }}

            /* ================================================================
               PREDICT BUTTON
               ================================================================ */
            div.stButton > button {{
                background: linear-gradient(135deg, {HONEY} 0%, {HONEY_DARK} 100%);
                color: {DEEP_BROWN} !important;
                font-weight: 700;
                border: none;
                border-radius: 10px;
                padding: 0.55rem 1.4rem;
                transition: transform 0.08s ease-in-out, box-shadow 0.15s;
                box-shadow: 0 2px 8px rgba(244,163,0,0.30);
            }}
            div.stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 16px rgba(244,163,0,0.45);
                color: {DEEP_BROWN} !important;
            }}

            /* ================================================================
               RESULT CARD
               ================================================================ */
            .result-card {{
                background: linear-gradient(
                    135deg,
                    {theme['result_bg_a']} 0%,
                    {theme['result_bg_b']} 100%
                );
                border: 1px solid rgba(244,163,0,0.40);
                border-radius: 14px;
                padding: 1.5rem 1.8rem;
                margin: 1rem 0 1.4rem 0;
                text-align: center;
            }}
            .result-card .label {{
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1.8px;
                color: {theme['accent']} !important;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }}
            .result-card .genus {{
                font-family: 'Poppins', sans-serif;
                font-size: 2.2rem;
                font-weight: 700;
                color: {theme['result_genus']} !important;
                margin-bottom: 0.25rem;
            }}
            .result-card .confidence {{
                font-size: 1.05rem;
                color: {theme['result_conf']} !important;
            }}
            .result-card .confidence strong {{
                color: {HONEY} !important;
            }}

            /* ================================================================
               FOOTER
               ================================================================ */
            .app-footer {{
                margin-top: 3rem;
                padding-top: 1.2rem;
                border-top: 1px solid {theme['card_border']};
                text-align: center;
                font-size: 0.85rem;
                color: {theme['footer_color']} !important;
            }}
            .app-footer a {{
                color: {HONEY} !important;
                font-weight: 600;
                text-decoration: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    theme = get_theme_colors()

    # --------------------------------------------------------------------
    # Img preprocessing
    # --------------------------------------------------------------------
    def preprocess_img(beeImgFile, IMG_SIZE=416):
        rawImg = keras.utils.load_img(beeImgFile, target_size=(IMG_SIZE, IMG_SIZE))
        imgArr = keras.utils.array_to_img(rawImg)
        imgArrExp = np.expand_dims(imgArr, 0)
        beeImg = tf.keras.applications.mobilenet_v3.preprocess_input(imgArrExp)
        return beeImg

    def load_model(model_file):
        model = joblib.load(model_file)
        return model

    def display_predictions(preds, labels):
        genus = []
        genusPreds = []
        for index, pred in enumerate(preds.flatten()):
            genus.append(labels[str(index)])
            genusPreds.append(round(pred * 100, 2))

        pred_df = pd.DataFrame()
        pred_df['Genus'] = genus
        pred_df['Pred'] = genusPreds
        pred_df = pred_df.sort_values('Pred', ascending=True)

        top_idx = pred_df['Pred'].idxmax()
        top_genus = pred_df.loc[top_idx, 'Genus']
        top_conf = pred_df.loc[top_idx, 'Pred']

        # --- Top-result highlight card ---
        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Most likely genus</div>
                <div class="genus">🐝 {top_genus}</div>
                <div class="confidence">Confidence: <strong>{top_conf:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Theme-aware chart colours ---
        text_color = "#FFF8E7" if theme["is_dark"] else "#1A1000"
        grid_color = "rgba(255,255,255,0.07)" if theme["is_dark"] else "rgba(0,0,0,0.07)"
        bar_low = "#7a4f14" if theme["is_dark"] else "#f5d8a0"
        bar_high = HONEY

        fig = go.Figure(
            go.Bar(
                x=pred_df['Pred'],
                y=pred_df['Genus'],
                orientation='h',
                marker=dict(
                    color=pred_df['Pred'],
                    colorscale=[[0, bar_low], [1, bar_high]],
                ),
                text=[f"{v:.1f}%" for v in pred_df['Pred']],
                textposition='outside',
                textfont=dict(color=text_color, size=12),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Prediction Confidence by Genus",
            title_font=dict(family="Poppins, sans-serif", size=18, color=text_color),
            xaxis_title="Probability (%)",
            yaxis_title="",
            height=520,
            margin=dict(l=10, r=55, t=60, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color),
            xaxis=dict(
                gridcolor=grid_color,
                range=[0, max(100, pred_df['Pred'].max() * 1.18)],
                color=text_color,
            ),
            yaxis=dict(color=text_color),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View full results table"):
            st.dataframe(
                pred_df.sort_values('Pred', ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

    def run_prediction(beeImgFile):
        """Shared prediction routine used by both tabs."""
        with st.spinner(text='🔬 Identification in progress... please wait.'):
            model = load_model('smoteImgBees17genus_classification_v3_model_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
            labels = joblib.load('smoteImgBees17genus_classification_v3_LABELS_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
            testImgPreds = model.predict(beeImgFile)
            display_predictions(testImgPreds, labels)

    # --------------------------------------------------------------------
    # Page config + styling
    # --------------------------------------------------------------------
    st.set_page_config(
        page_title="NBDC: Bee Identification",
        page_icon="🐝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css(theme)

    # --------------------------------------------------------------------
    # Hero header
    # --------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-subtitle">NATIONAL BEE DIAGNOSTIC CENTRE · NORTHWESTERN POLYTECHNIC</div>
            <div class="hero-title">🐝 Bee Genus Identification</div>
            <div class="hero-tagline">
                Upload a photo of a bee specimen and our MobileNetV3-based model will predict its genus,
                along with a confidence breakdown across all 17 recognized genera.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------------------
    with st.sidebar:
        st.image("nbdc-bees.jpg", caption="NBDC Bees", use_container_width=True)

        st.markdown(
            """
            <div class="sidebar-card">
                <h5>About this tool</h5>
                <p>This tool uses a <strong>MobileNetV3</strong> deep learning model to classify
                bee images into one of 17 genera. Upload a clear, well-lit image of a single
                specimen for best results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <h5>How to upload</h5>
                <p>Choose one of two methods:</p>
                <ol>
                    <li><strong>Upload a file</strong> &mdash; JPG, PNG, or JPEG from your device.</li>
                    <li><strong>Image URL</strong> &mdash; paste a direct link to an image.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <h5>Resources</h5>
                <p>
                    🔗 <a href="https://www.nwpolytech.ca/research/national-bee-diagnostic-centre" target="_blank">About NBDC</a><br>
                    🎬 <a href="https://youtu.be/yQKpkCMbHxY" target="_blank">Video walkthrough</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------
    # Tab state tracking
    # --------------------------------------------------------------------
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None

    st.markdown("### 📤 Choose an upload method")

    tab_file, tab_url = st.tabs(["📁  Upload File", "🔗  Image URL"])

    # ---------------- TAB 1 - FILE UPLOAD ----------------
    with tab_file:
        st.markdown("#### Upload an image from your device")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse for a JPG, PNG, or JPEG file",
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
        )

        if uploaded_file is not None:
            st.session_state.active_tab = "file"

            with st.spinner(text='Loading image... please wait'):
                image = Image.open(uploaded_file)

            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col_info:
                st.success("✅ Image uploaded successfully!")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
                predict_clicked = st.button("🔍 Predict from file", key="predict_file")

            if uploaded_file is not None and 'predict_clicked' in locals() and predict_clicked:
                if st.session_state.active_tab != "file":
                    st.warning("The URL tab is currently active. Clear the URL first to use file upload.")
                else:
                    try:
                        beeImgFile = preprocess_img(uploaded_file)
                        run_prediction(beeImgFile)
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
        else:
            if st.session_state.active_tab == "file":
                st.session_state.active_tab = None
            st.info("👆 Please upload an image file to get started.")

    # ---------------- TAB 2 - IMAGE URL ----------------
    with tab_url:
        st.markdown("#### Provide a direct image URL")
        url = st.text_input("Enter Image URL:", key="url_input", placeholder="https://example.com/bee.jpg")

        if url:
            st.session_state.active_tab = "url"

            try:
                response = requests.get(url)
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    with st.spinner(text='Loading image... please wait'):
                        image = Image.open(BytesIO(response.content))

                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        st.image(image, caption="Image from URL", use_container_width=True)
                    with col_info:
                        st.success("✅ Image loaded successfully!")
                        st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
                        predict_clicked_url = st.button("🔍 Predict from URL", key="predict_url")

                    if predict_clicked_url:
                        if st.session_state.active_tab != "url":
                            st.warning("The file upload tab is currently active. Remove the uploaded file first to use URL.")
                        else:
                            try:
                                beeImgFile = preprocess_img(BytesIO(response.content))
                                run_prediction(beeImgFile)
                            except Exception as e:
                                st.error(f"Error in prediction: {e}")
                else:
                    st.error("The URL does not point to a valid image. Content-Type received was " + content_type)

            except requests.HTTPError as e:
                st.error(f"HTTP Error occurred: {str(e)}")
            except requests.RequestException as e:
                st.error(f"Failed to fetch image due to request exception: {str(e)}")
        else:
            if st.session_state.active_tab == "url":
                st.session_state.active_tab = None
            st.info("👆 Please enter an image URL to get started.")

    # --------------------------------------------------------------------
    # Footer
    # --------------------------------------------------------------------
    st.markdown(
        """
        <div class="app-footer">
            Built for the <strong>National Bee Diagnostic Centre</strong> ·
            <a href="https://www.nwpolytech.ca/research/national-bee-diagnostic-centre" target="_blank">Northwestern Polytechnic</a>
            &nbsp;|&nbsp; Powered by MobileNetV3
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
