import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from model import load_model


CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag','Other']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("rice_model.pth", device)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# UI
st.title("Rice Grain Classifier")
st.write("Upload a rice grain image and the model will identify its variety.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=700)


    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]

    predicted_class = CLASS_NAMES[probabilities.argmax().item()]
    confidence = probabilities.max().item() * 100


# Results
    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")

    st.markdown("#### All class probabilities:")
    for name, prob in zip(CLASS_NAMES, probabilities):
        st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")