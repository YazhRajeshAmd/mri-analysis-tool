# MRI Analysis Tool

🧠 **AI-Powered Medical Imaging Analysis with Deep Learning and LLM Insights**

A comprehensive medical imaging analysis tool that combines advanced computer vision, deep learning, and Large Language Model (LLM) capabilities to analyze MRI scans and provide professional medical insights.

## 🎯 Features

- **Multi-format Support**: 
  - DICOM files (.dcm)
  - NIfTI files (.nii, .nii.gz)
  - Standard image formats (PNG, JPG, etc.)
- **Advanced Image Processing**:
  - Noise reduction and filtering
  - Contrast enhancement
  - Region of interest (ROI) analysis
  - 3D volume rendering
- **AI-Powered Analysis**: 
  - Deep learning-based feature extraction
  - K-means clustering for tissue segmentation
  - LLM-generated medical reports using Llama 3.1 70B
- **Interactive Web Interface**: Professional Gradio-based medical UI
- **Comprehensive Reporting**: Detailed analysis reports with measurements and findings

## 🛠️ Technology Stack

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision and image processing
- **NiBabel & PyDICOM**: Medical imaging file formats
- **LangChain + Ollama**: LLM integration for medical analysis
- **scikit-learn**: Machine learning algorithms
- **Gradio**: Professional medical interface
- **NumPy & SciPy**: Scientific computing

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** and pull the Llama 3.1 70B model:
   ```bash
   ollama pull llama3.1:70b
   ```

2. **Python 3.8+** with CUDA support (optional, for GPU acceleration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mri-analysis-tool.git
   cd mri-analysis-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama server:
   ```bash
   ollama serve
   ```

4. Run the application:
   ```bash
   python mri.py
   ```

5. Open your browser and navigate to the provided Gradio URL

## 📊 Usage

### Image Upload & Analysis

1. **Upload MRI Scan**: Drag and drop DICOM, NIfTI, or image files
2. **Select Analysis Type**: 
   - Brain MRI analysis
   - Spine MRI analysis
   - Custom region analysis
3. **Configure Parameters**:
   - Contrast adjustment
   - Noise reduction level
   - Segmentation sensitivity
4. **Generate Report**: AI-powered medical analysis with findings

### Analysis Features

- **Tissue Segmentation**: Automatic identification of different tissue types
- **Measurement Tools**: Distance, area, and volume calculations
- **3D Visualization**: Volume rendering for complex structures
- **Comparative Analysis**: Multi-timepoint comparison capabilities

## 🎯 Analysis Capabilities

### Image Processing
- **Preprocessing**: Denoising, normalization, bias field correction
- **Enhancement**: Contrast optimization, histogram equalization
- **Segmentation**: K-means clustering, region growing, edge detection

### Medical Analysis
- **Anatomical Structure Detection**: Automatic identification of key structures
- **Pathology Detection**: Anomaly identification and highlighting
- **Quantitative Analysis**: Volume measurements, intensity analysis
- **Report Generation**: Professional medical reports with LLM insights

## 🔧 Configuration

### Model Settings
```python
# LLM Configuration
llm = Ollama(model="llama3.1:70b", temperature=0.1)

# Image Processing Parameters
NOISE_REDUCTION_KERNEL = (5, 5)
CONTRAST_ENHANCEMENT = True
SEGMENTATION_CLUSTERS = 4
```

### Supported Formats
- **DICOM**: `.dcm`, `.dicom`
- **NIfTI**: `.nii`, `.nii.gz`
- **Standard**: `.png`, `.jpg`, `.jpeg`, `.tiff`

## 📁 Project Structure

```
mri/
├── mri.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gradio/           # Gradio configuration
└── models/            # Pre-trained models (optional)
```

## 🔬 Technical Details

### Deep Learning Components
- **Feature Extraction**: Convolutional neural networks for image analysis
- **Segmentation**: U-Net style architectures for tissue segmentation
- **Classification**: ResNet-based models for pathology detection

### Medical Imaging Standards
- **DICOM Compliance**: Full support for medical imaging standards
- **Coordinate Systems**: RAS/LPS coordinate system handling
- **Metadata Preservation**: Patient information and scan parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/medical-feature`)
3. Commit your changes (`git commit -m 'Add medical feature'`)
4. Push to the branch (`git push origin feature/medical-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This tool is for research and educational purposes only. It is NOT intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for medical advice and diagnosis.

## 🔗 Related Projects

- [Financial Stock Analysis](https://github.com/yourusername/fsi-stock-analysis)
- [ROCm RAG Assistant](https://github.com/yourusername/rocm-rag-assistant)

## 📚 References

- [Medical Image Computing and Computer Assisted Intervention](https://www.miccai.org/)
- [DICOM Standard](https://www.dicomstandard.org/)
- [NIfTI Format Specification](https://nifti.nimh.nih.gov/)

---

**Built with ❤️ for advancing medical AI research**
