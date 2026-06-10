# Rice Grain Classifier

A deep learning model that identifies 5 varieties of rice from an image, with out-of-distribution rejection for non-rice inputs.

**[Live Demo](YOUR_STREAMLIT_LINK)** | **[Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)**

---

## Classes
| Variety | Description |
|---------|-------------|
| Arborio | Short-grain, used in risotto |
| Basmati | Long-grain, aromatic |
| Ipsala | Medium-grain, grown in Turkey |
| Jasmine | Long-grain, fragrant |
| Karacadag | Short-grain, grown in Turkey |

---

## How It Works

1. Upload a rice grain image
2. Model predicts the variety with confidence score
3. If the image is not a rice grain, the model rejects it instead of giving a wrong prediction

---

## Tech Stack

- **Model:** ResNet18 with transfer learning (PyTorch)
- **Training:** 82K images (75K rice + 6.9K random images for rejection)
- **Deployment:** Streamlit Community Cloud
- **Out-of-distribution handling:** Entropy-based rejection + confidence threshold

---

## Project Structure

```
rice-classifier/
├── app.py                  # Streamlit application
├── model.py                # Model architecture and loading
├── requirements.txt        # Dependencies
├── rice_model.pth          # Trained model weights
└── sample_images/          # Test images for each variety
    ├── arborio.jpg
    ├── basmati.jpg
    ├── ipsala.jpg
    ├── jasmine.jpg
    └── karacadag.jpg
```

---

## Sample Images

Sample test images for all 5 varieties are available in the `sample_images` folder. Download any of them to test the live demo.

---

## Training Details

| Detail | Value |
|--------|-------|
| Base model | ResNet18 (ImageNet pretrained) |
| Dataset size | 82,000 images |
| Classes | 6 (5 rice varieties + Other) |
| Input size | 224×224 |
| Optimizer | Adam (lr=0.0001) |
| Epochs | 10 |

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/rice-classifier
cd rice-classifier
pip install -r requirements.txt
streamlit run app.py
```

---

## What I Learned

- Transfer learning with ResNet18 on a custom dataset
- Handling class imbalance with weighted loss
- Out-of-distribution detection using entropy
- End-to-end deployment with Streamlit