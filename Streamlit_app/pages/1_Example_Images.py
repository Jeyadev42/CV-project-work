import streamlit as st
from PIL import Image
import os

st.title("Example Images for Anomaly Detection")
st.write("Use the following examples if you don't have an image to test. You can download them and upload on the main detection page.")

image_dir = "example"

example_images = {
    "Bottle": {
        "Good": ["Bottle1.png", "Bottle2.png", "Bottle3.png", "Bottle4.png"],
        "Defect": ["Bottle-BrokenLarge1.png", "Bottle-BrokenLarge2.png",
                   "Bottle-Contamination1.png", "Bottle-Contamination2.png"]
    },
    "Capsule": {
        "Good": ["Capsule1.png", "Capsule2.png", "Capsule3.png", "Capsule4.png"],
        "Defect": ["Capsule-Scratch1.png", "Capsule-Scratch2.png",
                   "Capsule-Squeeze1.png", "Capsule-Squeeze2.png"]
    },
    "Leather": {
        "Good": ["Leather1.png", "Leather2.png", "Leather3.png", "Leather4.png"],
        "Defect": ["Leather-glue1.png", "Leather-glue2.png",
                   "Leather-poke1.png", "Leather-poke2.png"]
    }
}

for category, types in example_images.items():
    st.header(f"{category}")
    for condition, filenames in types.items():
        st.subheader(f"{condition} Samples")
        cols = st.columns(4)
        for i, filename in enumerate(filenames):
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                with cols[i % 4]:
                    st.image(Image.open(img_path), caption=filename, use_container_width=True)
                    with open(img_path, "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f,
                            file_name=filename,
                            mime="image/png",
                            key=f"{category}_{condition}_{filename}"
                        )
            else:
                st.error(f"❌ File not found: {img_path}")
