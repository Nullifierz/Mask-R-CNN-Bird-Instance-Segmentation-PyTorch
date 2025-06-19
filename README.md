# Mask R-CNN Bird Detection Project

## 📋 Project Description

This project uses **Mask R-CNN** with ResNet-50 FPN architecture for real-time bird detection and segmentation. The model is fine-tuned using the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset with PyTorch and TorchVision.

### 🎯 Key Features
- **Object Detection**: Detect birds in images/videos with bounding boxes
- **Instance Segmentation**: Provide precise masks/segmentation for each detected bird
- **Fine-tuning**: Pre-trained COCO model adapted for bird classification
- **Checkpoint Management**: Flexible model saving and loading system

## 🗂️ Project Structure

```
Fine-tuning/
├── Mask RCNN.ipynb              # Main notebook for training and evaluation
├── README.md                    # Project documentation
├── mask_rcnn_burung_final.pth   # Final trained model
├── model/                       # Model checkpoint folder
│   ├── mask_rcnn_burung_epoch_*.pth
│   └── mask_rcnn_burung_final_checkpoint.pth
├── CUB_200_2011/               # Caltech-UCSD Birds dataset
│   ├── images/                 # Bird images
│   ├── bounding_boxes.txt      # Bounding box coordinates
│   └── images.txt              # Image filename list
├── segmentations/              # Manual segmentation masks
├── test/                       # Test images
└── maskrcnn_venv/              # Python virtual environment
```

## 🛠️ Installation and Setup

### 1. My Specification System
- Python 3.11.11
- NVIDIA GeForce RTX 4060 8GB VRAM
- CUDA v12.9 and CUDNN v9.10

### 2. Dependencies Installation

```bash
# Activate virtual environment
cd maskrcnn_venv/Scripts
activate.bat

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # You need to install it manual
pip install -r requirements.txt
```

### 3. Dataset Setup

1. Download CUB-200-2011 dataset and segmentations from [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
2. Extract to `CUB_200_2011/` and `segmentations/segmentations` folder
3. Ensure folder structure matches the description above

## 🚀 Usage

### 1. Model Training

Open `Mask RCNN.ipynb` in Jupyter Notebook and run cells sequentially:

```python
# 1. Import libraries and setup device
# 2. Load and modify Mask R-CNN model
# 3. Prepare dataset and dataloader
# 4. Training loop
# 5. Save checkpoint and final model
# 6. SINGLE IMAGE PREDICTION AND TESTING
```

#### Training Parameters:
- **Epochs**: 10 (adjustable)
- **Learning Rate**: 0.005
- **Batch Size**: 2
- **Optimizer**: SGD with momentum 0.9
- **Scheduler**: StepLR (reduce by 0.1 every 3 epochs)

### 2. Model Evaluation

```python
# Run evaluation cells in notebook for:
# - Visualization of prediction vs ground truth results
# - Testing on individual images
# - Model performance analysis
```

## 🏗️ Model Architecture

### Base Model: Mask R-CNN ResNet-50 FPN
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **RPN**: Region Proposal Network for object detection
- **ROI Head**: Fast R-CNN for classification and bounding box regression
- **Mask Head**: FCN for instance segmentation

### Modifications for Bird Detection:
```python
# Change number of classes to 2 (background + bird)
num_classes_birds = 2

# Replace classification head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes_birds)

# Replace segmentation head
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes_birds)
```

## 📊 Dataset Information

### Caltech-UCSD Birds-200-2011
- **Total Images**: 11,788 images
- **Bird Species**: 200 bird species
- **Annotations**: 
  - Bounding boxes
  - Part annotations
  - Attribute annotations
- **Split**: Training and testing splits provided in `train_test_split.txt`

### Custom Segmentation Masks
- Format: PNG grayscale
- Background: 0 (black)
- Foreground (bird): 255 (white)
- Location: `segmentations/segmentations/`

## 🔧 Configuration

### Model Parameters
```python
# Detection threshold
score_threshold = 0.7  # For evaluation
score_threshold = 0.5  # For real-time (more sensitive)

# Dataset paths
ROOT_DIR = 'CUB_200_2011'
SEGMENTATIONS_DIR = 'segmentations/segmentations'

# Checkpoint directory
checkpoint_dir = "model"
```

### Training Configuration
```python
# Optimizer settings
lr = 0.005
momentum = 0.9
weight_decay = 0.0005

# Scheduler settings
step_size = 3  # Reduce LR every 3 epochs
gamma = 0.1    # Multiply LR by 0.1
```

## 📈 Performance Metrics

### Training Results
- **Final Training Loss**: ~0.5-0.8 (depending on epoch)
- **Training Time**: ~12-14 hours (with CUDA GPU)
- **Model Size**: ~167MB

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 1
   
   # Or use CPU
   device = torch.device("cpu")
   ```

2. **Model Not Found**
   ```python
   # Ensure correct model path
   model_load_path = "model/mask_rcnn_burung_final.pth"
   
   # Or use absolute path
   model_load_path = os.path.abspath("model/mask_rcnn_burung_final.pth")
   ```

4. **ImportError on Libraries**
   ```bash
   # Reinstall dependencies
   pip install --upgrade torch torchvision
   pip install --upgrade opencv-python
   ```

## 📁 File Descriptions

### Core Files
- **`Mask RCNN.ipynb`**: Main notebook containing training, evaluation, and testing
- **`mask_rcnn_burung_final.pth`**: Final trained model weights

### Checkpoint Files
- **`mask_rcnn_burung_epoch_XX.pth`**: Complete checkpoint per epoch
- **`mask_rcnn_burung_model_only_epoch_XX.pth`**: Model weights only per epoch
- **`mask_rcnn_burung_final_checkpoint.pth`**: Final complete checkpoint

### Dataset Files
- **`bounding_boxes.txt`**: Bounding box coordinates for each image
- **`images.txt`**: Image ID to filename mapping
- **`train_test_split.txt`**: Training/testing data split

## 🔮 Future Improvements

1. **Model Enhancement**
   - Experiment with larger backbones (ResNet-101, ResNeXt)
   - Implement advanced data augmentation
   - Multi-scale training and testing

2. **Dataset Expansion**
   - Add more bird species variations
   - Data augmentation for different lighting scenarios
   - More precise segmentation annotations

3. **Deployment**
   - Model optimization for inference (TensorRT, ONNX)
   - Web app implementation with Flask/FastAPI
   - Mobile deployment with PyTorch Mobile

4. **Features**
   - Bird species classification (multi-class)
   - Multiple bird tracking in video
   - Bird behavior analysis

## 👥 Contributors

- **Developer**: Aris Marcel Luis
- **Dataset**: Caltech-UCSD Birds-200-2011
- **Framework**: PyTorch, TorchVision

## 📄 License

This project uses the CUB-200-2011 dataset which is available for academic and research purposes. Please ensure compliance with the dataset license when using for commercial purposes.

---

**Happy Bird Detecting! 🐦**
