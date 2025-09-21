# Deepfake Detection using CNN & Frequency Domain Analysis

An **AI-powered Deepfake Detection System** that identifies manipulated videos and images by combining **spatial features (CNNs)** with **frequency domain features (FFT/DCT)**.  
The project integrates **data preprocessing, model training with ResNet/Xception**, performance evaluation, and a **multi-page demo web application** with **Grad-CAM explainability**.

---

## 📌 Features
- Detects **deepfake images and videos** with high accuracy.  
- Uses **dual-branch architecture**:  
  - CNN branch for spatial domain features.  
  - Frequency branch (FFT/DCT) for hidden spectral artifacts.  
- Implements **transfer learning models** (ResNet, XceptionNet).  
- Includes **Grad-CAM explainability** to highlight manipulated regions.  
- Provides a **multi-page web app** for uploading and testing media.  
- Supports **evaluation metrics**: Accuracy, Precision, Recall, F1-score, and ROC-AUC.  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Deepfake-Detection-using-CNN-Frequency-Domain-Analysis.git
cd Deepfake-Detection-using-CNN-Frequency-Domain-Analysis

2. Install Dependencies
pip install -r requirements.txt

3. Dataset Preparation

Download datasets such as FaceForensics++ or DFDC.

Extract frames using preprocessing scripts.

Apply face detection (MTCNN/dlib) and frequency transforms (FFT/DCT).

4. Model Training

Train CNN and transfer learning models:

python training/train_model.py --model resnet --epochs 20
python training/train_model.py --model xception --epochs 20

5. Run the Web App
streamlit run app/main.py


or (if Flask backend):

python app.py

📊 Evaluation Metrics

Accuracy: Overall correctness of predictions.

Precision: How many predicted fakes are actually fake.

Recall: How many real fakes the model successfully detected.

F1-Score: Balance between Precision & Recall.

ROC-AUC: Measures classification performance across thresholds.

🔎 Explainability

We use Grad-CAM (Gradient-weighted Class Activation Mapping) to generate heatmaps showing which image regions influenced the model’s decision, making the system more interpretable and trustworthy.

🛠️ Tech Stack

Programming: Python

Deep Learning: TensorFlow / PyTorch, Keras

Computer Vision: OpenCV, Dlib, MTCNN

Models: CNN, ResNet, XceptionNet

Signal Processing: NumPy, SciPy (FFT/DCT)

Deployment: Streamlit / Flask

Visualization: Matplotlib, Seaborn, Grad-CAM

🌍 Applications

Media Verification – Detecting fake news & manipulated content.

Cybersecurity – Preventing fraud & scams.

Forensics – Authenticating digital evidence.

Social Media – Automatic fake content filtering.

Education & Research – Advancing AI-based content authentication.

📈 Future Scope

Extend detection to audio deepfakes.

Enable real-time detection for live streams.

Integrate with social media APIs for automatic flagging.

Deploy lightweight versions for mobile and edge devices.

👥 Team Members

Stuti Attri (10322210071) – Data Collection & Preprocessing

Ishani Nag (10322210075) – Model Training & Evaluation

Saniya Shaikh (10322210086) – Application & Integration

📚 References

Rossler, A., et al. FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV, 2019.

Dolhansky, B., et al. The DeepFake Detection Challenge Dataset. arXiv:2006.07397, 2020.

Li, Y., & Lyu, S. Exposing DeepFake Videos by Detecting Face Warping Artifacts. CVPRW, 2019.

Wang, S.Y., et al. CNN-Generated Images Are Surprisingly Easy to Spot... for Now. CVPR, 2020.

Kaggle Datasets – Deepfake Detection Dataset, 2021.