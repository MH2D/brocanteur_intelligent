import streamlit as st
from PIL import Image
import io
import joblib
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

DATA_PATH = './data/'
model_name = 'transfert_learning_test_acc_0.22.pt'
model = torch.load(DATA_PATH + model_name, map_location=torch.device('cpu'))
model.eval() 

# Define the image preprocessing transform
img_size = (258, 258)
preprocess = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# Define the class labels for the ImageNet dataset (ResNet-18 output)
class_labels = joblib.load(f'{DATA_PATH}all_kamera_store_models.sav')

def classify_image(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_labels[predicted_idx.item()]

    return predicted_label

def main():
    st.title("Image Classification App")
    st.write("Upload an image and get the label of the object in the photo.")
    st.markdown(model_name)

    taken_photo = st.camera_input("Take a picture")
    # taken_photo = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if taken_photo is not None:
        # Display the uploaded image
        image = Image.open(taken_photo)
        # st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform image classification
        predicted_label = classify_image(image)

        # Show the label of the object
        st.write("Label:", predicted_label)

if __name__ == "__main__":
    main()
