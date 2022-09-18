import streamlit as st
import pathlib
import plotly.express as px
temb = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from fastai.vision.all import *

# title
st.title("Transportni klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=(['png','jpg','jpeg','gif','svg']))

if file is not None:
    img = Image.open(file)
    st.image(img, caption = 'Yuklangan rasm')
    if file:
      #st.image(file)
      #PIL convert
      img = PILImage.create(file)
      #model
      model = load_learner('technologies_model.pkl')

      # prediction
      pred, pred_id, probs = model.predict(img)
      st.success(f'Prognoz:{pred}')
      st.info(f'Ehtimollik:{probs[pred_id]*100: .1f}%')

      #plotting
      fig = px.bar(x=probs*100, y=model.dls.vocab)
      st.plotly_chart(fig)
else:
    print("Iltimos, modelga  mos rasm yuklang")
