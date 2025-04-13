import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import models
import joblib
import os
import tempfile
from azure.storage.blob import BlobServiceClient
from model import (
    CapsulePatchAutoencoder,
    ProductClassifier
)

# ----------------- SETUP -----------------
device = "cpu"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@st.cache_resource(show_spinner=False)
def download_blob_from_azure(container: str, blob_name: str, local_filename: str):
    connection_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_str)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
    local_path = os.path.join(tempfile.gettempdir(), local_filename)
    if not os.path.exists(local_path):
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
    return local_path

with st.spinner(" Loading models from Azure Blob Storage..."):
    classifier_path = download_blob_from_azure("model-dump", "product_classifier.pth", "product_classifier.pth")
    capsule_path = download_blob_from_azure("model-dump", "model_capsule_patch.pth", "model_capsule_patch.pth")
    bottle_feat_path = download_blob_from_azure("model-dump", "resnet_bottle_train_features.npy", "resnet_bottle_train_features.npy")
    leather_feat_path = download_blob_from_azure("model-dump", "resnet_leather_train_features.npy", "resnet_leather_train_features.npy")
    bottle_nn_path = download_blob_from_azure("model-dump", "resnet_bottle_nn_model.joblib", "resnet_bottle_nn_model.joblib")
    leather_nn_path = download_blob_from_azure("model-dump", "resnet_leather_nn_model.joblib", "resnet_leather_nn_model.joblib")

# Load classifier
classifier = ProductClassifier().to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()

# Capsule autoencoder
model_capsule = CapsulePatchAutoencoder().to(device)
model_capsule.load_state_dict(torch.load(capsule_path, map_location=device))
model_capsule.eval()

# ResNet setup
from torchvision.models import ResNet18_Weights
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
resnet.eval()
features = []
resnet.layer2.register_forward_hook(lambda m, i, o: features.append(o))

# Load features and models
train_feats = {
    "bottle": np.load(bottle_feat_path),
    "leather": np.load(leather_feat_path)
}

nn_models = {
    "bottle": joblib.load(bottle_nn_path),
    "leather": joblib.load(leather_nn_path)
}

# ----------------- HELPERS -----------------

def predict_category(tensor, classifier):
    with torch.no_grad():
        tensor = tensor.unsqueeze(0).to(device)
        logits = classifier(tensor)
        pred = torch.argmax(logits, dim=1).item()
        label_map = {"bottle": 0, "capsule": 1, "carpet": 2, "hazelnut": 3, "leather": 4}
        if pred in label_map.values():
            pred = list(label_map.keys())[list(label_map.values()).index(pred)]
        else:
            pred = "unknown"
        return pred

def extract_resnet_feature(img):
    features.clear()
    _ = resnet(img.unsqueeze(0).to(device))
    return features[0].squeeze().cpu().detach().numpy()

def compute_anomaly_map(test_feat, nn_model):
    test_flat = test_feat.reshape(test_feat.shape[0], -1).T
    dists, _ = nn_model.kneighbors(test_flat)
    return dists[:, 0].reshape(test_feat.shape[1], test_feat.shape[2])

def overlay_heatmap(original, error_map):
    original = np.array(original.resize((256, 256)))
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    elif original.shape[2] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    error_norm = ((error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay

def detect_capsule_anomaly(image_tensor, model, patch_size=128, stride=64):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    _, _, H, W = image_tensor.shape
    error_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint8)
    with torch.no_grad():
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = image_tensor[:, :, i:i+patch_size, j:j+patch_size]
                recon = model(patch)
                patch_error = torch.abs(patch - recon).mean(dim=1).squeeze().cpu().numpy()
                error_map[i:i+patch_size, j:j+patch_size] += patch_error
                count_map[i:i+patch_size, j:j+patch_size] += 1
    return error_map / np.maximum(count_map, 1)

def resnet_detector(tensor, category_predicted, img):
    test_feat = extract_resnet_feature(tensor)
    error_map = compute_anomaly_map(test_feat, nn_models[category_predicted])
    overlay = overlay_heatmap(img, error_map)
    overlay_resized = cv2.resize(overlay, (300, 300))
    img_resized = img.resize((300, 300))
    score = np.mean(error_map) * 100
    return overlay_resized, img_resized, score

def capsule_autoencoder(tensor, img):
    error_map = detect_capsule_anomaly(tensor, model_capsule)
    overlay = overlay_heatmap(img, error_map)
    overlay_resized = cv2.resize(overlay, (300, 300))
    img_resized = img.resize((300, 300))
    score = np.mean(error_map) * 100
    return overlay_resized, img_resized, score

# ----------------- UI -----------------
st.title("Anomaly Detector")
st.subheader("Detect defects in products using deep learning models")
st.info("Note: Anomaly detection is currently only available for bottle, leather, and capsule categories.")
st.info("You can use the example images from the sidebar if you don't have an image to test.")
st.write("Upload an image of a product to check for defects.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    tensor = transform(img)

    category_predicted = predict_category(tensor, classifier)
    st.info(f"Predicted Category: **{category_predicted.upper()}**")

    supported_categories = ["bottle", "leather", "capsule"]
    if category_predicted in supported_categories:
        st.success(f"✅ Running anomaly detection for {category_predicted.upper()}")
        if category_predicted in ["bottle", "leather"]:
            overlay_resized, img_resized, score = resnet_detector(tensor, category_predicted, img)
        elif category_predicted == "capsule":
            overlay_resized, img_resized, score = capsule_autoencoder(tensor, img)

        threshold = 10
        st.write(f"Anomaly Score: {score:.2f}%")
        if score > threshold:
            st.error(f"⚠️ This is a Defective {category_predicted.upper()}.")
        else:
            st.success(f"✅ This is a GOOD {category_predicted.upper()}.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Original Image", use_container_width =False)
        with col2:
            st.image(overlay_resized, caption="Anomaly Heatmap", channels="BGR", use_container_width =False)
    else:
        st.warning(f"⚠️ Anomaly detection is not yet available for `{category_predicted.upper()}`.")
