import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DISABLE_MKL"] = "1"
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import tf_keras as keras
import plotly.graph_objects as go
import folium
import folium.plugins
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from streamlit_gsheets import GSheetsConnection


# ----------------------------------------------------------------------------
# THEME / COLOUR PALETTE
# ----------------------------------------------------------------------------
HONEY = "#F4A300"
HONEY_DARK = "#C9810A"
DEEP_BROWN = "#2B1D0E"

# ----------------------------------------------------------------------------
# DISTRIBUTION MAP CONFIG
# Sightings (genus + optional GPS coords) are stored in a Google Sheet rather
# than local disk — Streamlit Community Cloud's filesystem is ephemeral and
# wipes local files on every redeploy/restart, so a CSV on disk wouldn't
# survive. Requires a `[connections.gsheets]` block in .streamlit/secrets.toml
# (service-account credentials) and a worksheet tab named below with header
# row: timestamp, genus, confidence, latitude, longitude, source.
# See: https://github.com/streamlit/gsheets-connection
# ----------------------------------------------------------------------------
GSHEETS_WORKSHEET = "Sightings"
SIGHTINGS_COLUMNS = ["timestamp", "genus", "confidence", "latitude", "longitude", "source"]
ALBERTA_CENTER = {"lat": 55.0, "lon": -115.0}

# A distinct marker color per genus for the Distribution Map's points layer
# (Folium's built-in named marker palette — order matches the 17 genera the
# classifier recognizes).
GENERA_17 = [
    "Agapostemon", "Andrena", "Anthidium", "Anthophora", "Apis", "Bombus",
    "Coelioxys", "Colletes", "Diadasia", "Halictus", "Hoplitis", "Hylaeus",
    "Lasioglossum", "Megachile", "Melissodes", "Osmia", "Xeromelecta",
]
GENUS_MARKER_COLORS = [
    "red", "blue", "green", "purple", "orange", "darkred", "lightred",
    "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "pink",
    "lightblue", "lightgreen", "gray", "black",
]
GENUS_COLOR_MAP = dict(zip(GENERA_17, GENUS_MARKER_COLORS))

# ----------------------------------------------------------------------------
# GENUS REFERENCE INFO
# Concise factual summaries compiled from public bee-identification sources
# (Minnesota Native Bees, Wikipedia, Project Dragonfly, Exotic Bee ID, etc).
# Edit freely — these are meant as an editable starting point, not final copy.
# Image files are expected at: images/<Genus>.jpg  (e.g. images/Halictus.jpg)
# ----------------------------------------------------------------------------
GENUS_INFO = {
    "Halictus": (
        "Halictus, commonly called sweat bees, belong to the family Halictidae "
        "and include over 200 species found mainly across the Northern "
        "Hemisphere. They are small to medium-sized, typically dark brown to "
        "black and sometimes with a metallic green sheen, with pale hair bands "
        "along the outer edge of each abdominal segment. Most nest in burrows "
        "in the ground and some species show primitively social behaviour, "
        "living in small colonies with overlapping generations. They are "
        "sometimes attracted to human sweat, which gives the group its common name."
    ),
    "Lasioglossum": (
        "Lasioglossum is the largest genus of bees in the world, with well over "
        "1,800 described species. Members are typically tiny to medium-sized "
        "and dusky black, brown, dull green, or blue, with hair bands set along "
        "the inner edge of the abdominal segments rather than the outer edge as "
        "in the closely related Halictus. Most nest in the ground, and social "
        "behaviour within the genus is extremely variable, ranging from solitary "
        "nesting to small eusocial colonies. Because of their small size they are "
        "often the most overlooked bees in a given habitat despite being among "
        "the most abundant."
    ),
    "Agapostemon": (
        "Agapostemon, known as metallic green sweat bees, are medium-sized bees "
        "in the family Halictidae notable for their bright, often brilliant "
        "metallic green head and thorax. In several species the males have a "
        "green head and thorax paired with a black-and-yellow striped abdomen, "
        "making them visually distinctive among native bees. They nest in the "
        "ground, sometimes in aggregations where many females share a single "
        "nest entrance while maintaining separate brood cells, and are common, "
        "easily recognized pollinators across much of North America."
    ),
    "Megachile": (
        "Megachile, the leafcutter bees, belong to the family Megachilidae and "
        "are recognized by their habit of cutting neat, circular or oval pieces "
        "from leaves and petals to line and partition their nest cells. They are "
        "dark gray to black, small to large bees with broad bodies and large "
        "mandibles used for cutting plant material. Unlike many bees, females "
        "carry pollen on a dense brush of hair (scopa) on the underside of the "
        "abdomen rather than on the hind legs. Most species nest in pre-existing "
        "cavities such as hollow stems, and the introduced alfalfa leafcutter "
        "bee (Megachile rotundata) is widely managed for crop pollination."
    ),
    "Colletes": (
        "Colletes, commonly called cellophane or plasterer bees, belong to the "
        "family Colletidae and are medium-sized, densely hairy, ground-nesting "
        "bees. They take their common name from the thin, cellophane-like "
        "secretion females apply to line their underground brood cells, which "
        "waterproofs the nest and helps keep liquid larval provisions from "
        "leaking out. Many species nest in dense aggregations of individual "
        "burrows in sandy or bare soil, and some emerge very early in spring, "
        "making them important early-season pollinators."
    ),
    "Melissodes": (
        "Melissodes, the long-horned bees, are medium-sized bees in the family "
        "Apidae with robust, broad abdomens. Males are easily recognized by "
        "their unusually long antennae, while females have long pollen-collecting "
        "hairs on their hind legs. Most species are pollen specialists on plants "
        "in the sunflower family (Asteraceae), and they are typically active from "
        "midsummer into autumn, making them common visitors to late-season "
        "composite flowers such as sunflowers and asters."
    ),
    "Apis": (
        "Apis is the genus of true honey bees, of which the Western honey bee "
        "(Apis mellifera) is by far the most familiar and economically important "
        "species worldwide. Honey bees are medium-sized, golden-brown and black "
        "bees that live in large, highly organized perennial colonies with a "
        "single queen, female workers, and seasonal males (drones). Unlike most "
        "native bees, Apis is eusocial and builds wax comb, storing honey and "
        "pollen for year-round survival. Managed honey bee colonies are widely "
        "used for commercial crop pollination in addition to their role as wild "
        "and feral pollinators."
    ),
    "Xeromelecta": (
        "Xeromelecta is a small genus of cuckoo bees in the family Apidae. Like "
        "other cuckoo bees, they are kleptoparasites: females do not build their "
        "own nests or collect pollen, but instead enter the nests of host bees "
        "(often digger bees in the genus Anthophora) to lay their eggs, with the "
        "Xeromelecta larva consuming the host's stored pollen provisions. As is "
        "typical of cuckoo bees, they lack the dense pollen-carrying hairs seen "
        "in pollen-collecting genera and instead have a sparser, often "
        "wasp-like appearance."
    ),
    "Osmia": (
        "Osmia, the mason bees, are stocky, often brilliantly metallic blue, "
        "green, or purple bees in the family Megachilidae. Like leafcutter bees, "
        "females carry pollen on a scopa beneath the abdomen rather than on the "
        "hind legs. They are named for their nesting habit of using mud or other "
        "masticated plant material to construct and seal partitions between "
        "brood cells, typically within pre-existing cavities such as hollow "
        "stems or holes in wood. Several Osmia species, including the orchard "
        "mason bee, are valued and sometimes commercially managed as efficient "
        "early-spring orchard pollinators."
    ),
    "Diadasia": (
        "Diadasia, sometimes called chimney bees or cactus bees, are robust, "
        "ground-nesting bees in the family Apidae. Many species build a short "
        "turret or 'chimney' of soil around their nest entrance, the function of "
        "which is not fully understood but may help protect the burrow from "
        "weather or predators. Diadasia are often pollen specialists, with "
        "different species associated with particular plant groups such as "
        "mallows, cacti, or globemallows, and they frequently nest in dense "
        "aggregations in open, sandy ground."
    ),
    "Hoplitis": (
        "Hoplitis, a genus of mason bees in the family Megachilidae, are small "
        "to medium bees, generally non-metallic apart from a few vividly green "
        "species. They have light blue-green eyes, a moderately pitted body, and "
        "white hair bands on the abdomen that are characteristically interrupted "
        "on the first two segments. Both sexes have a distinctive curling "
        "(concave) abdomen, and males often have unusual hooked or pointed "
        "antennae. Females collect pollen on hairs beneath the abdomen and nest "
        "in pre-existing cavities such as plant stems, lining brood cells with "
        "chewed leaf material mixed with plant pith."
    ),
    "Hylaeus": (
        "Hylaeus, the masked or yellow-faced bees, are very small, nearly "
        "hairless bees in the family Colletidae that can be mistaken for small "
        "wasps. They are best identified by pale yellow or white facial markings, "
        "which are typically more extensive in males than females. Unusually for "
        "bees, Hylaeus lack external pollen-carrying hairs altogether; instead, "
        "females ingest pollen and nectar and carry the mixture internally in "
        "their crop back to the nest. They nest above ground in narrow, "
        "pre-existing cavities such as hollow or pith-filled plant stems."
    ),
    "Coelioxys": (
        "Coelioxys, the sharp-tailed cuckoo bees, are kleptoparasites in the "
        "family Megachilidae that target the nests of leafcutter bees (Megachile) "
        "and their relatives. Females are recognized by a pointed, cone-shaped "
        "tip to the abdomen used to lay eggs inside a host's sealed brood cell, "
        "while males typically have several spines or teeth at the abdomen's "
        "end. As with other cuckoo bees, Coelioxys do not collect pollen "
        "themselves and lack scopal hairs, relying instead on the food stores "
        "left by their unwitting hosts."
    ),
    "Bombus": (
        "Bombus, the bumble bees, are large, robust, densely hairy bees in the "
        "family Apidae, easily recognized by their bold black-and-yellow (or "
        "sometimes orange or white) banded coloration. They are eusocial, "
        "forming annual colonies of a single queen and up to roughly 100 "
        "workers, often nesting in abandoned rodent burrows, thick grass, or "
        "other insulated cavities. Bumble bees are capable of 'buzz "
        "pollination,' vibrating flowers to release pollen that other bees "
        "cannot access, making them especially important pollinators of crops "
        "such as tomatoes, blueberries, and many wildflowers."
    ),
    "Anthophora": (
        "Anthophora, commonly called digger bees, are fast-flying, often densely "
        "hairy bees in the family Apidae that nest in burrows in the ground or "
        "in soft rock and cliff faces. Many species are notably robust and "
        "bee-fly-like in flight, hovering rapidly near flowers. Anthophora often "
        "nest in dense aggregations, and some species favor steep banks or "
        "vertical soil faces for their burrows. They are generalist or "
        "near-generalist foragers and frequent visitors to a wide range of "
        "spring and summer flowers."
    ),
    "Anthidium": (
        "Anthidium, the wool carder bees, are rotund, medium-sized bees in the "
        "family Megachilidae with distinctive yellow-and-black banded "
        "coloration that can resemble wasps at a glance. Their common name "
        "comes from the females' habit of scraping soft hairs ('wool') from the "
        "leaves of plants such as lamb's ear, which they carry back to line "
        "their nest cells in pre-existing cavities. Males of several Anthidium "
        "species are notably territorial, aggressively patrolling and defending "
        "patches of flowers from other male bees and insects."
    ),
    "Andrena": (
        "Andrena, the mining bees, form one of the largest bee genera with "
        "well over 1,500 described species worldwide and is especially diverse "
        "in temperate regions. They are small to medium-sized, often hairy bees, "
        "typically black or with a dull metallic blue or green cast, and "
        "females usually carry pollen on dense hairs along the hind legs. "
        "Andrena are solitary ground-nesters, with each female excavating her "
        "own burrow, though nests are frequently found in loose aggregations. "
        "Many species emerge early in spring and are important pollinators of "
        "early-blooming trees, shrubs, and wildflowers."
    ),
}

GENUS_FALLBACK = (
    "Detailed reference information for this genus hasn't been added yet. "
    "Check back soon, or contact the NBDC team if you'd like to contribute "
    "information for this species."
)



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
               TABS  (st.tabs, used for nothing now but kept for safety)
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
               RADIO-AS-TABS  (st.radio styled to look and feel like tabs,
               used instead of st.tabs because Streamlit's st.tabs has no
               way to be switched programmatically from a button click —
               st.radio's selection CAN be driven via session_state.)
               ================================================================ */
            div[role="radiogroup"] {{
                gap: 8px;
                border-bottom: 1px solid {theme['card_border']};
                padding-bottom: 0;
                margin-bottom: 1rem;
            }}
            div[role="radiogroup"] label {{
                background-color: {theme['tab_hover_bg']};
                border-radius: 10px 10px 0 0 !important;
                padding: 10px 22px !important;
                font-weight: 600;
                margin-bottom: 0 !important;
                border-bottom: 3px solid transparent;
            }}
            div[role="radiogroup"] label[data-checked="true"],
            div[role="radiogroup"] label:has(input:checked) {{
                background-color: rgba(244,163,0,0.18) !important;
                border-bottom: 3px solid {HONEY} !important;
            }}
            div[role="radiogroup"] label:has(input:checked) p {{
                color: {theme['accent']} !important;
                font-weight: 700 !important;
            }}
            /* Hide the default radio circle so it reads purely as a tab */
            div[role="radiogroup"] label > div:first-child {{
                display: none;
            }}

            /* ================================================================
               GENUS INFO TAB
               ================================================================ */
            .genus-info-card {{
                background: {theme['card_bg']};
                border: 1px solid {theme['card_border']};
                border-radius: 16px;
                padding: 1.6rem 1.8rem;
                margin-bottom: 1rem;
            }}
            .genus-info-title {{
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
                font-size: 1.8rem;
                color: {theme['result_genus']} !important;
                margin-bottom: 0.6rem;
            }}
            .genus-info-body {{
                font-size: 1rem;
                line-height: 1.7;
                color: {theme['sidebar_text']} !important;
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


def load_sightings_log():
    """Load the persisted specimen-sighting log from the Google Sheet.

    Returns an empty, correctly-columned DataFrame if the sheet is empty,
    unreachable, or not yet configured — callers never need to special-case it.
    """
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=GSHEETS_WORKSHEET, ttl=5)
        df = df.dropna(how="all")
        for col in SIGHTINGS_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[SIGHTINGS_COLUMNS]
    except Exception as e:
        st.warning(f"Couldn't reach the sightings sheet: {e}")
        return pd.DataFrame(columns=SIGHTINGS_COLUMNS)


def append_sighting(genus, confidence, latitude, longitude, source):
    """Append one identification (with its optional geolocation) to the sheet."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing = conn.read(worksheet=GSHEETS_WORKSHEET, ttl=0)
        existing = existing.dropna(how="all")
        new_row = pd.DataFrame([{
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "genus": genus,
            "confidence": confidence,
            "latitude": latitude,
            "longitude": longitude,
            "source": source,
        }])
        updated = pd.concat([existing, new_row], ignore_index=True)
        conn.update(worksheet=GSHEETS_WORKSHEET, data=updated)
        st.cache_data.clear()
    except Exception as e:
        st.warning(f"Couldn't log this sighting to the sheet: {e}")


def main():
    theme = get_theme_colors()

    # --------------------------------------------------------------------
    # Img preprocessing
    # --------------------------------------------------------------------
    def preprocess_img(beeImgFile, IMG_SIZE=416):
        # Reset file pointer in case it was already read (Streamlit uploads)
        if hasattr(beeImgFile, 'seek'):
            beeImgFile.seek(0)
        rawImg = keras.utils.load_img(beeImgFile, target_size=(IMG_SIZE, IMG_SIZE))
        imgArr = keras.utils.img_to_array(rawImg)
        imgArrExp = np.expand_dims(imgArr, 0)
        beeImg = tf.keras.applications.mobilenet_v3.preprocess_input(imgArrExp)
        return beeImg

    def load_model(model_file):
        model = joblib.load(model_file)
        return model

    def display_predictions(preds, labels, latitude=None, longitude=None, source=""):
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
        st.session_state.predicted_genus = top_genus

        if latitude is not None and longitude is not None:
            append_sighting(top_genus, float(top_conf), float(latitude), float(longitude), source)
            st.caption(f"📍 Logged at ({latitude:.4f}, {longitude:.4f}) — see it on the Distribution Map tab.")

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

        def go_to_genus_tab():
            st.session_state.active_section_idx = 2  # index of "📖  Genus Info"

        col_spacer, col_btn, col_spacer2 = st.columns([1, 1.4, 1])
        with col_btn:
            st.button(
                f"📖 Want to know more about {top_genus}?",
                key="learn_more_btn",
                on_click=go_to_genus_tab,
                width="stretch",
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
        st.plotly_chart(fig, width="stretch")

        with st.expander("📋 View full results table"):
            st.dataframe(
                pred_df.sort_values('Pred', ascending=False).reset_index(drop=True),
                width="stretch",
                hide_index=True,
            )

    def run_prediction(beeImgFile, latitude=None, longitude=None, source=""):
        """Shared prediction routine used by both tabs."""
        try:
            with st.spinner(text='🔬 Identification in progress... please wait.'):
                model = load_model('smoteImgBees17genus_classification_v3_model_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
                labels = joblib.load('smoteImgBees17genus_classification_v3_LABELS_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
                testImgPreds = model.predict(beeImgFile, verbose=0)
                display_predictions(testImgPreds, labels, latitude=latitude, longitude=longitude, source=source)
        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}")
        except Exception as e:
            import traceback
            st.error(f"Prediction failed: {e}")
            print("FULL ERROR:", traceback.format_exc())

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
        st.image("nbdc-bees.jpg", caption="NBDC Bees", width="stretch")

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

        # QR code — scan to open the app on any device
        if os.path.exists("qr_code.png"):
            st.markdown(
                """
                <div class="sidebar-card" style="text-align:center;">
                    <h5>Scan to open</h5>
                    <p style="margin-bottom:0.5rem;">Open this app on your phone or share with others</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image("qr_code.png", width="stretch")

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
    if "predicted_genus" not in st.session_state:
        st.session_state.predicted_genus = None
    if "active_section_idx" not in st.session_state:
        st.session_state.active_section_idx = 0

    st.markdown("### 📤 Choose an upload method")

    section_options = ["📁  Upload File", "🔗  Image URL", "📖  Genus Info", "🗺️  Distribution Map"]

    selected_display = st.radio(
        "Choose a section",
        section_options,
        index=st.session_state.active_section_idx,
        horizontal=True,
        label_visibility="collapsed",
    )

    # Update index based on what user manually clicked
    st.session_state.active_section_idx = section_options.index(selected_display)

    if "Upload File" in selected_display:
        current_section = "Upload File"
    elif "Image URL" in selected_display:
        current_section = "Image URL"
    elif "Genus Info" in selected_display:
        current_section = "Genus Info"
    else:
        current_section = "Distribution Map"

    # ---------------- TAB 1 - FILE UPLOAD ----------------
    if current_section == "Upload File":
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
                st.image(image, caption="Uploaded Image", width="stretch")
            with col_info:
                st.success("✅ Image uploaded successfully!")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")

                with st.expander("📍 Add collection location (optional)"):
                    record_loc_file = st.checkbox(
                        "Plot this specimen on the Distribution Map",
                        value=False,
                        key="loc_include_file",
                    )
                    loc_lat_file = st.number_input(
                        "Latitude", value=ALBERTA_CENTER["lat"], format="%.5f",
                        min_value=-90.0, max_value=90.0, key="loc_lat_file",
                    )
                    loc_lon_file = st.number_input(
                        "Longitude", value=ALBERTA_CENTER["lon"], format="%.5f",
                        min_value=-180.0, max_value=180.0, key="loc_lon_file",
                    )

                predict_clicked = st.button("🔍 Predict from file", key="predict_file")

            if uploaded_file is not None and 'predict_clicked' in locals() and predict_clicked:
                if st.session_state.active_tab != "file":
                    st.warning("The URL tab is currently active. Clear the URL first to use file upload.")
                else:
                    try:
                        beeImgFile = preprocess_img(uploaded_file)
                        run_prediction(
                            beeImgFile,
                            latitude=loc_lat_file if record_loc_file else None,
                            longitude=loc_lon_file if record_loc_file else None,
                            source=uploaded_file.name,
                        )
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
        else:
            if st.session_state.active_tab == "file":
                st.session_state.active_tab = None
            st.info("👆 Please upload an image file to get started.")

    # ---------------- TAB 2 - IMAGE URL ----------------
    elif current_section == "Image URL":
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
                        st.image(image, caption="Image from URL", width="stretch")
                    with col_info:
                        st.success("✅ Image loaded successfully!")
                        st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")

                        with st.expander("📍 Add collection location (optional)"):
                            record_loc_url = st.checkbox(
                                "Plot this specimen on the Distribution Map",
                                value=False,
                                key="loc_include_url",
                            )
                            loc_lat_url = st.number_input(
                                "Latitude", value=ALBERTA_CENTER["lat"], format="%.5f",
                                min_value=-90.0, max_value=90.0, key="loc_lat_url",
                            )
                            loc_lon_url = st.number_input(
                                "Longitude", value=ALBERTA_CENTER["lon"], format="%.5f",
                                min_value=-180.0, max_value=180.0, key="loc_lon_url",
                            )

                        predict_clicked_url = st.button("🔍 Predict from URL", key="predict_url")

                    if predict_clicked_url:
                        if st.session_state.active_tab != "url":
                            st.warning("The file upload tab is currently active. Remove the uploaded file first to use URL.")
                        else:
                            try:
                                beeImgFile = preprocess_img(BytesIO(response.content))
                                run_prediction(
                                    beeImgFile,
                                    latitude=loc_lat_url if record_loc_url else None,
                                    longitude=loc_lon_url if record_loc_url else None,
                                    source=url,
                                )
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

    # ---------------- TAB 3 - GENUS INFO ----------------
    elif current_section == "Genus Info":
        genus = st.session_state.predicted_genus

        if not genus:
            st.info("👆 Run an identification first, then come back here to learn more about your bee's genus.")
        else:
            info_text = GENUS_INFO.get(genus, GENUS_FALLBACK)
            image_path = f"images/{genus}.jpg"

            col_img, col_text = st.columns([1, 1.6])
            with col_img:
                if os.path.exists(image_path):
                    st.image(image_path, caption=genus, width="stretch")
                else:
                    st.info(f"📷 Add an image at `images/{genus}.jpg` to display it here.")
            with col_text:
                st.markdown(
                    f"""
                    <div class="genus-info-card">
                        <div class="genus-info-title">🐝 {genus}</div>
                        <div class="genus-info-body">{info_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---------------- TAB 4 - DISTRIBUTION MAP ----------------
    else:
        st.markdown("#### 🗺️ Wild Bee Genus Distribution — Alberta / Canada")
        st.caption(
            "Built from every identification where a user opted to log its collection "
            "coordinates (see the '📍 Add collection location' option on the upload tabs)."
        )

        sightings_df = load_sightings_log()
        geo_df = sightings_df.copy()
        geo_df["latitude"] = pd.to_numeric(geo_df["latitude"], errors="coerce")
        geo_df["longitude"] = pd.to_numeric(geo_df["longitude"], errors="coerce")
        geo_df = geo_df.dropna(subset=["latitude", "longitude"])

        if geo_df.empty:
            st.info(
                "No geotagged identifications yet. Upload a specimen, expand "
                "**'📍 Add collection location (optional)'**, tick the checkbox, and set "
                "its coordinates — it'll show up here after the next prediction."
            )
        else:
            genus_options = sorted(geo_df["genus"].dropna().unique().tolist())
            col_filter, col_count = st.columns([2, 1])
            with col_filter:
                genus_filter = st.multiselect(
                    "Filter by genus (select one or more)", genus_options, default=[], key="map_genus_filter"
                )
            with col_count:
                st.metric("Geotagged records", len(geo_df))

            show_points = st.checkbox(
                "Include individual specimen markers as a toggleable layer", value=True, key="map_show_points"
            )

            if not genus_filter:
                st.info("Select one or more genera above to show data on the map.")
            else:
                plot_df = geo_df[geo_df["genus"].isin(genus_filter)]

                if plot_df.empty:
                    st.warning("No records for the selected genera yet.")
                else:
                    map_title = "All genera" if set(genus_filter) == set(genus_options) else ", ".join(genus_filter)
                    st.markdown(f"**Density heatmap — {map_title}** ({len(plot_df)} records)")

                    # --- Build the Folium map (Leaflet-based — no WebGL required) ---
                    m = folium.Map(location=[ALBERTA_CENTER["lat"], ALBERTA_CENTER["lon"]], zoom_start=5, tiles="OpenStreetMap")

                    heat_data = plot_df[["latitude", "longitude", "confidence"]].values.tolist()
                    heat_layer = folium.FeatureGroup(name="Density heatmap", show=True)
                    HeatMap(heat_data, radius=18, blur=22, min_opacity=0.3).add_to(heat_layer)
                    heat_layer.add_to(m)

                    if show_points:
                        points_layer = folium.FeatureGroup(name="Specimen points", show=False)
                        for _, row in plot_df.iterrows():
                            color = GENUS_COLOR_MAP.get(row["genus"], "gray")
                            popup_html = (
                                f"<b>Genus:</b> {row['genus']}<br>"
                                f"<b>Confidence:</b> {row['confidence'] * 100:.1f}%<br>"
                                f"<b>Timestamp:</b> {row['timestamp']}"
                            )
                            folium.CircleMarker(
                                location=[row["latitude"], row["longitude"]],
                                radius=4,
                                color=color,
                                fill=True,
                                fill_opacity=0.8,
                                popup=folium.Popup(popup_html, max_width=250),
                            ).add_to(points_layer)
                        points_layer.add_to(m)

                    folium.plugins.Fullscreen(position="topleft").add_to(m)
                    folium.LayerControl(position="topright", collapsed=False).add_to(m)

                    st_folium(m, width=None, height=560, returned_objects=[])

                    # --- Raw log + export ---
                    with st.expander("📋 View raw log / export"):
                        display_df = plot_df.sort_values("timestamp", ascending=False).reset_index(drop=True).copy()
                        display_df["confidence"] = display_df["confidence"].apply(lambda c: f"{float(c) * 100:.1f}%")
                        st.dataframe(display_df, width="stretch", hide_index=True)
                        st.download_button(
                            "⬇️ Download sightings CSV",
                            data=geo_df.to_csv(index=False).encode("utf-8"),
                            file_name="bee_sightings_log.csv",
                            mime="text/csv",
                        )

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
