import streamlit as st
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection
import torch
from torchvision import models
from io import BytesIO
import pytesseract

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(
    page_title="Auto NPR",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def instantiate_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path = "model/best (4).pt", force_reload=True)
    #model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='model/best (2).pt',force_reload=True)
    model.eval()
    model.conf = 0.5
    model.iou = 0.45
    return model

@st.cache_data(persist=True)
def get_success_message():
    return '✅ Download Successful !!'

def download_success():
    st.balloons()
    st.success(get_success_message())


top_image = Image.open('static/banner_top.png')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')

upload_path = "uploads/"
download_path = "downloads/"
model = instantiate_model()

st.image(main_image,use_column_width='auto')
st.title(' Automatic Number Plate Recognition 🚘🚙')
st.sidebar.image(top_image,use_column_width='auto')
st.sidebar.header('Input 🛠')
selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image", "Live Video Feed"])
st.sidebar.image(bottom_image,use_column_width='auto')

if selected_type == "Upload Image":
    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png","jpg","bmp","jpeg"])

    if uploaded_file is not None:
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working... 💫"):
            uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))

            with open(uploaded_image,'rb') as imge:
                img_bytes = imge.read()

            img = Image.open(io.BytesIO(img_bytes))
            results = model(img, size=640)
            results.render()
            for img in results.ims:
                img_base64 = Image.fromarray(img)
                img_base64.save(downloaded_image, format="JPEG")

            final_image = Image.open(downloaded_image)
            print("Opening ",final_image)
            st.markdown("---")
            st.image(final_image, caption='This is how your final image looks like 😉')
            
            count = 0
            CONFIDENCE_THRESHOLD = 0.5
            for result in results.xyxy[0]:
             if result[4] > CONFIDENCE_THRESHOLD and result[5] == 1.0:
              count += 1
              x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
              p = img[y1:y2, x1:x2]
              st.image(p, caption=f'Number Plate {count}')
              cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
              cv2.putText(img, f'Number plate {count}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

              text = pytesseract.image_to_string(p)
              st.text(f"Number {count}:")
              st.write(text)

            
            
            # Using PyTesseract to extract the characters from the number plate
            with st.spinner("Extracting characters from number plate..."):
                extracted_text = pytesseract.image_to_string(final_image)
                st.success(f"Extracted characters: {extracted_text}")
                
                
                
            with open(downloaded_image, "rb") as file:
                if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/jpg'
                                         ):
                        download_success()
                if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/jpeg'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/png'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/bmp'
                                         ):
                        download_success()
    else:
        st.warning('⚠ Please upload your Image 😯')


else:
    st.info('✨ The Live Feed from Web-Camera will take some time to load up 🎦')
    live_feed = st.checkbox('Start Web-Camera ✅')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if live_feed:
        while(cap.isOpened()):
            success, frame = cap.read()
            if success == True:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                img = Image.open(io.BytesIO(frame))
                model = instantiate_model()
                results = model(img, size=640)
                results.print()
                img = np.squeeze(results.render())
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                count = 0
                CONFIDENCE_THRESHOLD = 0.5
                for result in results.xyxy[0]:
                  if result[4] > CONFIDENCE_THRESHOLD and result[5] == 1.0:
                    count += 1
                    x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
                    p = img[y1:y2, x1:x2]
                    st.image(p, caption=f'Number Plate {count}')
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'Number plate {count}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    text = pytesseract.image_to_string(p)
                    st.text(f"Number {count}:")
                    st.write(text)

            
            
            # Using PyTesseract to extract the characters from the number plate
                
                    with st.spinner("Extracting characters from number plate..."):
                        extracted_text = pytesseract.image_to_string(img)
                        
                        st.success(f"Extracted characters: {extracted_text}")
               
                
                
            else:
                break
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            FRAME_WINDOW.image(frame)
        
    else:
        cap.release()
        cv2.destroyAllWindows()
        st.warning('⚠ The Web-Camera is currently disabled. 😯')

st.markdown("<br><hr><center>Made with ❤️ by <a href='rajgokul.it21@bitsathy.ac.in?subject=Automatic Number Plate Recognition WebApp!&body=Please specify the issue you are facing with the app.'><strong>RAJGOKUL</strong></a></center><hr>", unsafe_allow_html=True)
