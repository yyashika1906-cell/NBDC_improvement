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
  #Img preprocessing
  def preprocess_img(beeImgFile, IMG_SIZE=416):
    #st.info('Preprocessing...')
    rawImg = keras.utils.load_img(beeImgFile, target_size=(IMG_SIZE,IMG_SIZE))
    imgArr = keras.utils.array_to_img(rawImg)
    #rawImg=image.load_img(beeImgFile, target_size=(IMG_SIZE,IMG_SIZE))
    #imgArr=image.img_to_array(rawImg)
    imgArrExp = np.expand_dims(imgArr, 0)
    beeImg=tf.keras.applications.mobilenet_v3.preprocess_input(imgArrExp)
    return beeImg
    
  def load_model(model_file):
    model = joblib.load(model_file)
    return model

  def display_predictions(preds, labels):
    #species=[]
    #specPreds=[]
    genus=[]
    genusPreds=[]
    for index, pred in enumerate(preds.flatten()):
      #st.write(f'{labels[index]}: {pred:.2f}%')
      #species.append(labels[index])
      #specPreds.append(pred)
      #st.success(f"Genus {labels[str(index)]} is processed")
      genus.append(labels[str(index)])
      genusPreds.append(round(pred*100,2))
      
    pred_df=pd.DataFrame()
    #pred_df['Species']=species
    #pred_df['Pred']=specPreds
    pred_df['Genus']=genus # Changed from species to genus
    pred_df['Pred']=genusPreds # Changed from specPreds to genusPreds
    
    fig = px.bar(pred_df, x="Pred", y="Genus", orientation='h',height=500, width=800, title='Bees Identification', labels=dict(Pred="Probabilities for Species (%)"), hover_data=["Pred"])
    st.plotly_chart(fig)
    
  # Set up your web app
  st.set_page_config(page_title="NDBC: Bees Identification",page_icon="🐝", layout="wide", initial_sidebar_state="expanded")

  st.title('National Bee Diagnostic Centre (NBDC)')
  st.header('NDBC Bees Identification with MobileNetV3')

  st.sidebar.image("nbdc-bees.jpg", caption='NDBC Bees')
  link1='More about Northwestern Polytechnic’s [NBDC](https://www.nwpolytech.ca/research/national-bee-diagnostic-centre)'
  st.sidebar.markdown(link1,unsafe_allow_html=True)
  st.sidebar.info("This tool uses MobileNetV3 to identify bee species based on their images. You have an interface to upload images of types jpg, png, or jpeg.")
  st.sidebar.markdown("### You have 2 options to upload the image:")
  st.sidebar.write("1) as a file (from your PC or laptop).")
  st.sidebar.write("2) or enter its URL")
  st.sidebar.markdown("## Guide:")
  link2='How to use this? [Vide Guide](https://youtu.be/yQKpkCMbHxY)'
  st.sidebar.markdown(link2,unsafe_allow_html=True)
  
  # Upload image file
  uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
  url = st.text_input("Or enter Image URL:")
  

  if uploaded_file is not None:
    # Display the uploaded image
    with st.spinner(text='Image loading... Please wait'):
      image = Image.open(uploaded_file)
      st.image(image, caption='Uploaded Image', width=200)
      st.success("Image uploaded successfully!")
    
    # Button to make a prediction
    if st.button('Predict from file'):
      try:
        beeImgFile=preprocess_img(uploaded_file)
        st.write("")
        #st.write("Identification...")
        with st.spinner(text='Identification in progress... Please wait.'):
          model = load_model('smoteImgBees17genus_classification_v3_model_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
          labels=joblib.load('smoteImgBees17genus_classification_v3_LABELS_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
          #model = load_model('model1_v3_NDBC_Bees_8_8_25.pkl')
          #labels=joblib.load('model1_v3_NDBC_Bees_8_8_25_labels.pkl')
          testImgPreds=model.predict(beeImgFile)
          st.success(f"Top class is: {labels[str(testImgPreds[0].argmax())]}")
          display_predictions(testImgPreds,labels)
          
      except Exception as e:
          st.error(f"Error in prediction: {e}")

  
  elif url:
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
          # Display the uploaded image
          with st.spinner(text='Image loading... Please wait'):
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Uploaded Image', width=200)
            st.success("Image uploaded successfully!")
              # Button to make a prediction
          if st.button('Predict from URL'):
            try:
              beeImgFile=preprocess_img(BytesIO(response.content))
              st.write("")
              #st.write("Identification...")
              with st.spinner(text='Identification in progress... Please wait.'):
                model = load_model('smoteImgBees17genus_classification_v3_model_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
                labels=joblib.load('smoteImgBees17genus_classification_v3_LABELS_1FullLarge1270adasynCW_9_5_26_32_416.pkl')
                testImgPreds=model.predict(beeImgFile)
                st.success(f"Top class is: {labels[str(testImgPreds[0].argmax())]}")
                display_predictions(testImgPreds,labels)
                
            except Exception as e:
                st.error(f"Error in prediction: {e}")
            
        else:
            st.error("The URL does not point to a valid image. Content-Type received was " + content_type)
            
    except requests.RequestException as e:
        st.error(f"Failed to fetch image due to request exception: {str(e)}")
        
    except requests.HTTPError as e:
        st.error(f"HTTP Error occurred: {str(e)}")
    
  else:
    st.info("Please upload an image file.")


    
  
 

if __name__ == '__main__':

    main()


























