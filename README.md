# RetinaFace: Deep Learning Face Detection & Recognition

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/serengp/RetinaFace?style=social)](https://github.com/serengp/RetinaFace)

A cutting-edge, production-ready face detection and recognition system powered by deep learning. RetinaFace combines state-of-the-art facial detection with facial landmark extraction and face alignment, delivering exceptional performance even in crowded scenes and challenging conditions.

---

## 📋 Table of Contents

- [Features](#features)
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Functionality](#core-functionality)
  - [Face Detection](#face-detection)
  - [Facial Landmarks](#facial-landmarks)
  - [Face Alignment](#face-alignment)
  - [Face Recognition](#face-recognition)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Performance Highlights](#performance-highlights)
- [Integration with DeepFace](#integration-with-deepface)
- [Dependencies](#dependencies)
- [License](#license)
- [Citation](#citation)
- [Contributing](#contributing)

---

## ✨ Features

- **Advanced Face Detection**: Multi-task face detection with bounding box regression
- **Facial Landmarks Extraction**: Precise detection of 5 facial landmarks (both eyes, nose, both mouth corners)
- **Face Alignment**: Automatic alignment of detected faces for improved recognition accuracy (~1% improvement)
- **Crowd Robustness**: Superior performance in crowded scenes and complex backgrounds
- **Multiple Backend Support**: TensorFlow implementation with easy deployment options
- **Face Recognition**: Integration with ArcFace for end-to-end face verification and identification
- **Easy Integration**: Simple pip-installable package with intuitive Python API
- **Pre-trained Weights**: Out-of-the-box ready with state-of-the-art pre-trained models

---

## 🎯 Overview

RetinaFace is the face detection module of the **InsightFace** project, originally implemented in MXNet and re-implemented in TensorFlow. This project provides a simplified, pip-compatible version that maintains the high performance of the original implementation while being easier to integrate into modern Python applications.

### Technology Stack

- **Deep Learning Framework**: TensorFlow/Keras
- **Face Detection Architecture**: Multi-task Face Detection Network
- **Face Recognition Model**: ArcFace (when integrated with DeepFace)
- **Image Processing**: OpenCV
- **Visualization**: Matplotlib

---

## 🔧 Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Install RetinaFace

```bash
# Using pip
pip install retina-face

# Or using conda
conda install -c conda-forge retina-face
```

### Step 2: Install Optional Dependencies

```bash
# For face recognition pipeline
pip install deepface

# For image processing and visualization
pip install opencv-python matplotlib
```

### Step 3: Verify Installation

```python
from retinaface import RetinaFace
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

print("✓ All dependencies installed successfully!")
```

---

## 🚀 Quick Start

### Basic Face Detection

```python
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# Load image
img_path = "image.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces
resp = RetinaFace.detect_faces(img_path)

# Draw bounding boxes
for key in resp:
    face = resp[key]
    facial_area = face['facial_area']  # [x1, y1, x2, y2]
    cv2.rectangle(img_rgb, (facial_area[0], facial_area[1]), 
                  (facial_area[2], facial_area[3]), (255, 0, 0), 2)

# Display result
plt.imshow(img_rgb)
plt.axis('off')
plt.title('Detected Faces')
plt.show()
```

---

## 🎓 Core Functionality

### Face Detection

RetinaFace detects all faces in an image and returns their bounding boxes with high accuracy.

**API:**
```python
RetinaFace.detect_faces(img_path)
```

**Returns:**
```python
{
    "face_1": {
        "facial_area": [x1, y1, x2, y2],
        "landmarks": {
            "left_eye": [...],
            "right_eye": [...],
            "nose": [...],
            "mouth_left": [...],
            "mouth_right": [...]
        },
        "score": confidence_score
    }
}
```

---

### Facial Landmarks

Extract precise facial landmarks (eyes, nose, mouth) for each detected face.

```python
response = RetinaFace.detect_faces("image.jpg")

for key in response:
    face = response[key]
    landmarks = face['landmarks']
    
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    nose = landmarks['nose']
    mouth_left = landmarks['mouth_left']
    mouth_right = landmarks['mouth_right']
```

---

### Face Alignment

Automatically align detected faces based on facial landmarks, improving recognition accuracy.

```python
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Extract and align faces
aligned_faces = RetinaFace.extract_faces("image.jpg", align=True)

# Display aligned faces
for face in aligned_faces:
    plt.imshow(face)
    plt.axis('off')
    plt.show()
```

**Impact**: Face alignment typically improves face recognition accuracy by approximately 1%.

---

### Face Recognition

Perform face verification and identification using RetinaFace with ArcFace backend.

```python
from deepface import DeepFace
from retinaface import RetinaFace

# Verify if two faces belong to the same person
result = DeepFace.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
    model_name='ArcFace',
    detector_backend='retinaface'
)

print(f"Verified: {result['verified']}")
print(f"Distance: {result['distance']:.4f}")
print(f"Threshold: {result['threshold']:.4f}")
```

---

## 📚 Usage Examples

### Example 1: Detect Multiple Faces in a Group Photo

```python
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# Load group image
img = cv2.imread("celeb_group.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect all faces
response = RetinaFace.detect_faces("celeb_group.jpg")

# Count and visualize
print(f"Detected {len(response)} faces")
for idx, key in enumerate(response):
    face = response[key]
    x1, y1, x2, y2 = face['facial_area']
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_rgb, f"Face {idx+1}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(img_rgb)
plt.title(f"Detected {len(response)} Faces")
plt.show()
```

### Example 2: Extract and Process Aligned Faces

```python
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Extract aligned faces
faces = RetinaFace.extract_faces("group_photo.jpg", align=True)

# Create a grid of aligned faces
fig, axes = plt.subplots(1, len(faces), figsize=(15, 5))
for idx, face in enumerate(faces):
    axes[idx].imshow(face)
    axes[idx].set_title(f"Face {idx+1}")
    axes[idx].axis('off')
plt.tight_layout()
plt.show()
```

### Example 3: Face Verification (1:1 Comparison)

```python
from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# Image paths
img1_path = "person_a.jpg"
img2_path = "person_b.jpg"

# Verify faces
result = DeepFace.verify(
    img1_path,
    img2_path,
    model_name='ArcFace',
    detector_backend='retinaface'
)

# Visualize results
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Draw bounding boxes
for img_path, img_rgb in [(img1_path, img1_rgb), (img2_path, img2_rgb)]:
    resp = RetinaFace.detect_faces(img_path)
    for key in resp:
        x1, y1, x2, y2 = resp[key]['facial_area']
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(img1_rgb)
ax1.set_title("Image 1")
ax1.axis('off')

ax2.imshow(img2_rgb)
ax2.set_title("Image 2")
ax2.axis('off')

verification_status = "✓ Same Person" if result["verified"] else "✗ Different Persons"
plt.suptitle(f"{verification_status} (Distance: {result['distance']:.4f})", fontsize=14, fontweight='bold')
plt.show()

print(f"Verified: {result['verified']}")
print(f"Distance: {result['distance']:.4f}")
print(f"Threshold: {result['threshold']:.4f}")
```

---

## 📁 Project Structure

```
Ratina Face/
│
├── README.md                              # Documentation
├── ratinaFace_Face_Detection.ipynb        # Main notebook with examples
│
├── Sample Images/
│   ├── celeb_group.jpg                    # Group photo for multi-face detection
│   ├── A_Heard.jpg                        # Sample celebrity image 1
│   ├── Amber.jpg                          # Sample celebrity image 2
│   ├── female-celebrity.jpg               # Sample image
│   ├── jake-gyllenhaal.jpg                # Sample image
│   ├── jenifer.jpeg                       # Sample image
│   ├── superman.jpeg                      # Sample image
│   ├── Sweedney.jpeg                      # Sample image
│   └── sweeney1.jpeg                      # Sample image
│
└── .ipynb_checkpoints/                    # Jupyter checkpoints
```

---

## 🚀 Performance Highlights

| Metric | Performance |
|--------|-------------|
| **Detection Accuracy** | 99%+ on standard benchmarks |
| **Face Alignment Improvement** | ~1% accuracy boost for recognition |
| **Landmark Extraction** | 5 keypoints (both eyes, nose, mouth corners) |
| **Multi-face Detection** | Efficient detection in crowded scenes |
| **Processing Speed** | Real-time capable on modern hardware |

---

## 🔗 Integration with DeepFace

For an end-to-end face recognition pipeline, RetinaFace integrates seamlessly with **DeepFace**:

- **Face Detection**: RetinaFace
- **Face Recognition**: ArcFace
- **Use Case**: 1:1 Verification, 1:N Identification

```python
from deepface import DeepFace

# One-liner for complete pipeline
result = DeepFace.verify(
    img1_path, 
    img2_path,
    model_name='ArcFace',
    detector_backend='retinaface'
)
```

---

## 📦 Dependencies

```
retina-face          # Core face detection
deepface             # Face recognition (optional)
opencv-python        # Image processing
matplotlib           # Visualization
tensorflow           # Deep learning backend
numpy                # Numerical computing
```

### Installation from Requirements

```bash
pip install retina-face deepface opencv-python matplotlib
```

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 📖 Citation

If you use RetinaFace in your research, please cite the original papers:

```bibtex
@inproceedings{deng2019retinaface,
  title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irini and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5203--5212},
  year={2020}
}

@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/ImdataScientistSachin/Ratina-Face.git
cd Ratina-Face

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📝 Notes

- **Image Formats Supported**: JPG, JPEG, PNG, BMP
- **Python Version**: 3.6+
- **Memory Requirements**: Minimal (pre-trained models are optimized)
- **GPU Acceleration**: Supported for faster inference

---

## 🙏 Acknowledgments

- Original RetinaFace implementation by the InsightFace project
- TensorFlow re-implementation reference by Stanislas Bertrand
- ArcFace for robust face recognition

---

## 📧 Support & Contact

For issues, questions, or suggestions:
- 📋 Open an GitHub Issue
- 📧 Email: [imdatascientistsachin@gmail.com]
- 💬 Discussions: [GitHub Discussions link]

---

## 🔮 Future Enhancements

- [ ] Real-time face detection from webcam
- [ ] Batch processing utilities
- [ ] Face attributes classification (age, gender, emotion)
- [ ] Performance optimization for edge devices
- [ ] Multi-GPU support
- [ ] Extended documentation and tutorials

---

**Made with ❤️ for the computer vision community**

---

*Last Updated: November 2025*
