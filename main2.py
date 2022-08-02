import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from streamlit_option_menu import option_menu

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('model2.hdf5')
	return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction

model = load_model()


with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ['Klasifikasi', 'Dataset'],
        default_index = 0,
        
    )

if selected == 'Klasifikasi':
    st.title("Klasifikasi Gambar")
    st.text("Upload gambar kuas, sikat_gigi, atau sikat_pakaian untuk mulai klasifikasi gambar")

    uploaded_file = st.file_uploader("Drag and drop gambarmu disini", type="jpg")
    if (uploaded_file is not None):
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang di upload', use_column_width=True)
        st.write("")
        
        with st.spinner("Klasifikasi....."):
            pred = predict_class(np.asarray(image), model)
            time.sleep(1)
            Class_Names = ['kuas', 'sikat_gigi', 'sikat_pakaian']
            result = Class_Names[np.argmax(pred)]
            output = 'Gambar {}, jumlah persentase : {:.0f}%'.format (result, 100 * np.max(pred))
            st.success(output)

            Class_Names = ['kuas', 'sikat_gigi', 'sikat_pakaian']
            result = Class_Names[np.argmin(pred)]
            output = 'Gambar {}, jumlah persentase : {:.0f}%'.format (result, 100 * np.min(pred))
            st.success(output)
                    

if selected == 'Dataset':
    st.title("Dataset")
    st.text("Dataset yang digunakan merupakan hasil gambar pribadi")
    st.text("Dataset terdiri dari 3 Class yaitu: \nClass ke-1 : Kuas memiliki gambar sebanyak 120\nClass ke-2 : Sikat Gigi memiliki gambar sebanyak 115\nClass ke-3 : Sikat Pakaian memiliki gambar sebanyak 100")
  
    st.text("*Epoch*\nJumlah epoch yang digunakan 50")
    st.text("*Batch Size*\nJumlah batch_size 32")
