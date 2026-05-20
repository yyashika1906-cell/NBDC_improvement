import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import keras
import plotly.express as px


def main():
  # Img preprocessing
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

    fig = px.bar(pred_df, x="Pred", y="Genus", orientation='h', height=500,
                 width=800, title='Bees Identification',
                 labels=dict(Pred="Probabilities for Species (%)"),
                 hover_data=["Pred"])
    st.plotly_chart(fig)

  def run_prediction(beeImgFile):
    """Shared prediction routine used by both tabs."""
    with st.spinner(text='Identification in progress... Please wait.'):
      model = load_model('smoteImgBees17genus_classification_v3_model_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
      labels = joblib.load('smoteImgBees17genus_classification_v3_LABELS_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
      testImgPreds = model.predict(beeImgFile)
      st.success(f"Top class is: {labels[str(testImgPreds[0].argmax())]}")
      display_predictions(testImgPreds, labels)

  # Set up your web app
  st.set_page_config(page_title="NDBC: Bees Identification", page_icon="🐝",
                     layout="wide", initial_sidebar_state="expanded")

  st.title('National Bee Diagnostic Centre (NBDC)')
  st.header('NDBC Bees Identification with MobileNetV3')

  st.sidebar.image("nbdc-bees.jpg", caption='NDBC Bees')
  link1 = 'More about Northwestern Polytechnic’s [NBDC](https://www.nwpolytech.ca/research/national-bee-diagnostic-centre)'
  st.sidebar.markdown(link1, unsafe_allow_html=True)
  st.sidebar.info("This tool uses MobileNetV3 to identify bee species based on their images. You have an interface to upload images of types jpg, png, or jpeg.")
  st.sidebar.markdown("### You have 2 options to upload the image:")
  st.sidebar.write("1) as a file (from your PC or laptop).")
  st.sidebar.write("2) or enter its URL")
  st.sidebar.markdown("## Guide:")
  link2 = 'How to use this? [Vide Guide](https://youtu.be/yQKpkCMbHxY)'
  st.sidebar.markdown(link2, unsafe_allow_html=True)

  # ---- Track which tab is the "active" one so only one can run at a time ----
  if "active_tab" not in st.session_state:
    st.session_state.active_tab = None

  st.markdown("#### Choose one method to identify your bee:")

  # Two tabs: only one method's prediction can run at a time
  tab_file, tab_url = st.tabs(["📁  Upload File", "🔗  Image URL"])

  # =====================================================================
  # TAB 1 - FILE UPLOAD
  # =====================================================================
  with tab_file:
    uploaded_file = st.file_uploader("Upload an image",
                                     type=["jpg", "jpeg", "png"],
                                     key="file_uploader")

    if uploaded_file is not None:
      # Mark this tab as the active one and lock the other
      st.session_state.active_tab = "file"

      with st.spinner(text='Image loading... Please wait'):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=200)
        st.success("Image uploaded successfully!")

      if st.button('Predict from file', key="predict_file"):
        if st.session_state.active_tab != "file":
          st.warning("The URL tab is currently active. Clear the URL first to use file upload.")
        else:
          try:
            beeImgFile = preprocess_img(uploaded_file)
            st.write("")
            run_prediction(beeImgFile)
          except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
      # If the file is removed and this tab was active, release the lock
      if st.session_state.active_tab == "file":
        st.session_state.active_tab = None
      st.info("Please upload an image file in this tab.")

  # =====================================================================
  # TAB 2 - IMAGE URL
  # =====================================================================
  with tab_url:
    url = st.text_input("Enter Image URL:", key="url_input")

    if url:
      # Mark this tab as the active one and lock the other
      st.session_state.active_tab = "url"

      try:
        response = requests.get(url)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
          with st.spinner(text='Image loading... Please wait'):
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Uploaded Image', width=200)
            st.success("Image uploaded successfully!")

          if st.button('Predict from URL', key="predict_url"):
            if st.session_state.active_tab != "url":
              st.warning("The file upload tab is currently active. Remove the uploaded file first to use URL.")
            else:
              try:
                beeImgFile = preprocess_img(BytesIO(response.content))
                st.write("")
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
      # If the URL is cleared and this tab was active, release the lock
      if st.session_state.active_tab == "url":
        st.session_state.active_tab = None
      st.info("Please enter an image URL in this tab.")


if __name__ == '__main__':
    main()
