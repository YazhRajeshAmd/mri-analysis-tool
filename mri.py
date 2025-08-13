import numpy as np
import matplotlib.pyplot as plt
import cv2
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import nibabel as nib
import pydicom
from sklearn.cluster import KMeans
from scipy import ndimage
import time
import os
from datetime import datetime
import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings('ignore')

# Initialize Ollama for medical analysis reports
llm = Ollama(model="llama3.1:70b", temperature=0.1)

# Medical analysis prompt template
medical_analysis_prompt = PromptTemplate(
    input_variables=["image_type", "findings", "measurements", "patient_info", "technical_params"],
    template="""
You are a radiologist AI assistant analyzing MRI scans. Provide a professional medical analysis based on the following data:

Image Type: {image_type}
Technical Parameters: {technical_params}
Patient Information: {patient_info}

Quantitative Findings:
{findings}

Measurements:
{measurements}

Please provide:
1. CLINICAL IMPRESSION: Overall assessment of the scan
2. DETAILED FINDINGS: Systematic analysis of anatomical structures
3. QUANTITATIVE ANALYSIS: Interpretation of measurements and statistics
4. RECOMMENDATIONS: Suggested follow-up or additional imaging if needed
5. TECHNICAL QUALITY: Assessment of image quality and acquisition parameters

Important: This is for educational/research purposes. All clinical decisions should be made by qualified medical professionals.

ANALYSIS:
"""
)

medical_analysis_chain = LLMChain(llm=llm, prompt=medical_analysis_prompt, verbose=True)

class MRIProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_dicom(self, dicom_path):
        """Load DICOM file"""
        try:
            ds = pydicom.dcmread(dicom_path)
            image = ds.pixel_array.astype(np.float32)
            
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            metadata = {
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'body_part': getattr(ds, 'BodyPartExamined', 'Unknown'),
                'slice_thickness': getattr(ds, 'SliceThickness', 'Unknown'),
                'pixel_spacing': getattr(ds, 'PixelSpacing', 'Unknown'),
            }
            
            return image, metadata
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            return None, None
    
    def load_nifti(self, nifti_path):
        """Load NIfTI file"""
        try:
            img = nib.load(nifti_path)
            image_data = img.get_fdata()
            
            # Get middle slice for 3D volumes
            if len(image_data.shape) == 3:
                middle_slice = image_data.shape[2] // 2
                image = image_data[:, :, middle_slice]
            else:
                image = image_data
            
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            metadata = {
                'shape': image_data.shape,
                'affine': img.affine.tolist(),
                'header': str(img.header)[:200] + "..."
            }
            
            return image, metadata
        except Exception as e:
            print(f"Error loading NIfTI: {e}")
            return None, None
    
    def preprocess_image(self, image):
        """Preprocess MRI image"""
        if image is None:
            return None
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Gaussian filtering for noise reduction
        filtered = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return filtered
    
    def segment_brain_tissue(self, image):
        """Segment brain tissue using K-means clustering"""
        if image is None:
            return None, {}
        
        # Reshape image for clustering
        pixels = image.reshape((-1, 1))
        
        # Apply K-means clustering (typically 3-4 clusters for brain MRI)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape back to image dimensions
        segmented = labels.reshape(image.shape)
        
        # Calculate tissue volumes (as percentages)
        unique, counts = np.unique(labels, return_counts=True)
        total_pixels = len(pixels)
        
        tissue_stats = {}
        for i, (cluster, count) in enumerate(zip(unique, counts)):
            percentage = (count / total_pixels) * 100
            tissue_stats[f'Tissue_Cluster_{cluster}'] = {
                'pixel_count': count,
                'percentage': round(percentage, 2)
            }
        
        return segmented, tissue_stats
    
    def detect_anomalies(self, image):
        """Detect potential anomalies using statistical methods"""
        if image is None:
            return None, {}
        
        # Calculate image statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Define anomaly threshold (pixels beyond 2.5 standard deviations)
        threshold = mean_intensity + 2.5 * std_intensity
        anomalies = image > threshold
        
        # Find connected components
        labeled_anomalies, num_features = ndimage.label(anomalies)
        
        anomaly_stats = {
            'num_anomalous_regions': num_features,
            'anomalous_pixel_percentage': (np.sum(anomalies) / image.size) * 100,
            'mean_intensity': round(mean_intensity, 2),
            'std_intensity': round(std_intensity, 2),
            'intensity_threshold': round(threshold, 2)
        }
        
        return anomalies, anomaly_stats
    
    def calculate_measurements(self, image, pixel_spacing=None):
        """Calculate various measurements from the MRI"""
        if image is None:
            return {}
        
        measurements = {
            'image_dimensions': f"{image.shape[0]} x {image.shape[1]} pixels",
            'total_pixels': image.size,
            'mean_intensity': round(np.mean(image), 2),
            'max_intensity': int(np.max(image)),
            'min_intensity': int(np.min(image)),
            'intensity_std': round(np.std(image), 2),
            'signal_to_noise_ratio': round(np.mean(image) / (np.std(image) + 1e-8), 2)
        }
        
        if pixel_spacing and pixel_spacing != 'Unknown':
            try:
                spacing = float(pixel_spacing[0]) if isinstance(pixel_spacing, list) else float(pixel_spacing)
                measurements['pixel_spacing_mm'] = spacing
                measurements['physical_width_mm'] = round(image.shape[1] * spacing, 2)
                measurements['physical_height_mm'] = round(image.shape[0] * spacing, 2)
            except:
                pass
        
        return measurements
    
    def generate_visualization(self, original, preprocessed, segmented, anomalies):
        """Generate comprehensive visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original MRI', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Enhanced (CLAHE + Gaussian Filter)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Segmented tissue
        axes[1, 0].imshow(segmented, cmap='viridis')
        axes[1, 0].set_title('Tissue Segmentation (K-means)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Anomaly detection
        anomaly_overlay = np.zeros_like(original)
        anomaly_overlay[anomalies] = 255
        axes[1, 1].imshow(original, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(anomaly_overlay, cmap='Reds', alpha=0.3)
        axes[1, 1].set_title('Anomaly Detection (Red Overlay)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig

def process_mri_scan(file_path, patient_info="Not provided", image_type="MRI Brain"):
    """Main function to process MRI scan"""
    if file_path is None:
        return None, "Please upload an MRI file", {}, "", "", ""
    
    processor = MRIProcessor()
    start_time = time.time()
    
    try:
        # Determine file type and load accordingly
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.dcm':
            image, metadata = processor.load_dicom(file_path)
            pixel_spacing = metadata.get('pixel_spacing', None) if metadata else None
        elif file_extension in ['.nii', '.nii.gz']:
            image, metadata = processor.load_nifti(file_path)
            pixel_spacing = None
        else:
            # Try to load as regular image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            metadata = {'file_type': 'standard_image'}
            pixel_spacing = None
        
        if image is None:
            return None, "Error: Could not load the image file", {}, "", "", ""
        
        # Process the image
        preprocessed = processor.preprocess_image(image)
        segmented, tissue_stats = processor.segment_brain_tissue(preprocessed)
        anomalies, anomaly_stats = processor.detect_anomalies(preprocessed)
        measurements = processor.calculate_measurements(image, pixel_spacing)
        
        # Generate visualization
        viz_fig = processor.generate_visualization(image, preprocessed, segmented, anomalies)
        
        # Prepare data for AI analysis
        findings_text = f"""
        Tissue Segmentation Analysis:
        {tissue_stats}
        
        Anomaly Detection Results:
        {anomaly_stats}
        
        Image Metadata:
        {metadata}
        """
        
        measurements_text = f"""
        Image Measurements:
        {measurements}
        """
        
        technical_params = f"File type: {file_extension}, Processing time: {time.time() - start_time:.2f}s"
        
        # Generate AI analysis
        ai_analysis = medical_analysis_chain.run(
            image_type=image_type,
            findings=findings_text,
            measurements=measurements_text,
            patient_info=patient_info,
            technical_params=technical_params
        )
        
        # Prepare summary statistics
        summary_stats = {
            "Processing Time": f"{time.time() - start_time:.2f} seconds",
            "Image Dimensions": measurements.get('image_dimensions', 'Unknown'),
            "Number of Tissue Clusters": len(tissue_stats),
            "Anomalous Regions Detected": anomaly_stats.get('num_anomalous_regions', 0),
            "Signal-to-Noise Ratio": measurements.get('signal_to_noise_ratio', 'N/A'),
            "Mean Intensity": measurements.get('mean_intensity', 'N/A')
        }
        
        return (
            viz_fig,
            "✅ MRI Analysis Completed Successfully",
            summary_stats,
            ai_analysis,
            f"Tissue Analysis: {tissue_stats}",
            f"Anomaly Detection: {anomaly_stats}"
        )
        
    except Exception as e:
        return None, f"❌ Error processing MRI: {str(e)}", {}, "", "", ""

# Create Gradio interface
def create_interface():
    # Custom CSS for AMD branding - fonts and colors only
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* AMD Red accent color */
    .primary {
        background: linear-gradient(135deg, #ED1C24 0%, #B71C1C 100%) !important;
        border: none !important;
    }
    
    .primary:hover {
        background: linear-gradient(135deg, #B71C1C 0%, #8B0000 100%) !important;
    }
    
    /* Tab styling with AMD red */
    .tab-nav button.selected {
        color: #ED1C24 !important;
        border-bottom: 2px solid #ED1C24 !important;
    }
    
    /* Headers with AMD red accents */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
    
    /* Input focus states with AMD red */
    input:focus, textarea:focus, select:focus {
        border-color: #ED1C24 !important;
        box-shadow: 0 0 0 2px rgba(237, 28, 36, 0.1) !important;
    }
    
    /* Links and accents */
    a {
        color: #ED1C24 !important;
    }
    
    /* Section headers */
    h3 {
        border-left: 4px solid #ED1C24 !important;
        padding-left: 12px !important;
    }
    """
    
    with gr.Blocks(title="🏥 Advanced MRI Analysis on AMD MI300X", theme=gr.themes.Soft(), css=custom_css) as interface:
        # Header with AMD logo in top right corner
        gr.HTML("""
            <div style="position: relative; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 15px; right: 20px; height: 35px; width: auto;" />
                <div style="padding-right: 120px;">
                    <h1 style="margin: 0; color: #2c3e50; font-size: 2.2em; font-weight: 700;">🏥 Advanced MRI Analysis System</h1>
                    <h3 style="margin: 5px 0 0 0; color: #ED1C24; font-size: 1.2em; font-weight: 600;">Powered by AMD MI300X GPU Acceleration</h3>
                </div>
            </div>
        """)
        
        gr.Markdown(
            """
            Upload MRI scans (DICOM, NIfTI, or standard image formats) for comprehensive AI-powered analysis including:
            - **Image Enhancement** with CLAHE and noise reduction
            - **Tissue Segmentation** using machine learning
            - **Anomaly Detection** with statistical analysis
            - **AI-Generated Medical Reports** using Large Language Models
            
            ⚠️ **Disclaimer**: This tool is for research and educational purposes only. Clinical decisions should always be made by qualified medical professionals.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload & Configuration")
                
                file_input = gr.File(
                    label="Upload MRI Scan",
                    file_types=[".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg"],
                    type="filepath"
                )
                
                patient_info = gr.Textbox(
                    label="Patient Information (Optional)",
                    placeholder="Age: 45, Gender: Male, Clinical History: Headaches...",
                    lines=3
                )
                
                image_type = gr.Dropdown(
                    choices=["MRI Abdomen", "MRI Brain",  "MRI Knee", "MRI Spine", "Other"],
                    label="Image Type",
                    value="MRI Abdomen"
                )
                
                analyze_btn = gr.Button("🔬 Analyze MRI Scan", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Analysis Results")
                
                status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ Visualizations"):
                        plot_output = gr.Plot(label="MRI Analysis Visualization")
                    
                    with gr.TabItem("📋 AI Medical Report"):
                        ai_analysis_output = gr.Textbox(
                            label="AI-Generated Medical Analysis",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.TabItem("📈 Technical Analysis"):
                        tissue_analysis_output = gr.Textbox(
                            label="Tissue Segmentation Results",
                            lines=8,
                            interactive=False
                        )
                        
                        anomaly_analysis_output = gr.Textbox(
                            label="Anomaly Detection Results", 
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.TabItem("📊 Summary Statistics"):
                        summary_output = gr.JSON(label="Processing Summary")
        
        # Event handlers
        analyze_btn.click(
            fn=process_mri_scan,
            inputs=[file_input, patient_info, image_type],
            outputs=[
                plot_output,
                status_output,
                summary_output,
                ai_analysis_output,
                tissue_analysis_output,
                anomaly_analysis_output
            ]
        )
        
        # Example section
        gr.Markdown(
            """
            ### 💡 Example Usage
            1. Upload an MRI scan file (DICOM .dcm, NIfTI .nii/.nii.gz, or standard image)
            2. Optionally provide patient information for context
            3. Select the appropriate image type
            4. Click "Analyze MRI Scan" to start processing
            5. Review the comprehensive analysis results across different tabs
            
            ### 🚀 AMD MI300X Acceleration
            This application leverages the powerful AMD MI300X GPU for:
            - Fast image processing and enhancement
            - Machine learning-based tissue segmentation
            - AI-powered medical report generation
            - Real-time anomaly detection algorithms
            """
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )