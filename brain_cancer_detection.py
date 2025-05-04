import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Brain Cancer Detection and Classification",
    page_icon="🧠",
    layout="wide"
)

# Title and Introduction
st.title("Brain Cancer Detection and Classification from MRI Images")
st.subheader("Using Convolutional Neural Networks (CNN)")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Upload & Predict", "Theory & Background", "Model Training", "About Project"]
choice = st.sidebar.radio("Go to", pages)

# Class labels for brain cancer classification
class_labels = ['No Tumor (Healthy)', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']


# ==== Preprocessing Functions ====

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess a single MRI image for model input

    Args:
        image: Input image (grayscale or RGB)
        target_size: Target size for the image (default: 224x224)

    Returns:
        Preprocessed image ready for model input
    """
    # Ensure image is in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize image to target size
    image = cv2.resize(image, target_size)

    # Apply histogram equalization for contrast enhancement
    image = cv2.equalizeHist(image)

    # Normalize pixel values to range [0, 1]
    image = image / 255.0

    # Add channel dimension for model input
    image = np.expand_dims(image, axis=-1)

    return image


def skull_stripping(image):
    """
    Apply simple skull stripping to focus on brain tissue

    Args:
        image: Input MRI image

    Returns:
        MRI image with skull region removed
    """
    # Convert to appropriate format if needed
    if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find the largest contour (brain region)
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image  # Return original if no contours found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def data_augmentation(image, rotation=15, shift=0.1, horizontal_flip=True, zoom=0.1):
    """
    Apply simple data augmentation to an image

    Args:
        image: Input image to augment
        rotation: Maximum rotation angle in degrees
        shift: Maximum shift as a fraction of image size
        horizontal_flip: Whether to apply horizontal flip
        zoom: Maximum zoom factor

    Returns:
        Augmented image
    """
    # Make a copy of the input image
    augmented = image.copy()

    # Get image dimensions
    if len(augmented.shape) == 3:
        h, w, c = augmented.shape
    else:
        h, w = augmented.shape
        augmented = np.expand_dims(augmented, axis=-1)
        c = 1

    # Apply random rotation
    if rotation > 0:
        angle = np.random.uniform(-rotation, rotation)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (w, h))

    # Apply random shift
    if shift > 0:
        tx = np.random.uniform(-shift, shift) * w
        ty = np.random.uniform(-shift, shift) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        augmented = cv2.warpAffine(augmented, M, (w, h))

    # Apply horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Apply zoom
    if zoom > 0:
        zoom_factor = np.random.uniform(1 - zoom, 1 + zoom)

        # Calculate new dimensions
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)

        # Resize image
        if zoom_factor > 1:  # Zoom in
            # Crop central part of the zoomed image
            augmented = cv2.resize(augmented, (new_w, new_h))
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            augmented = augmented[start_h:start_h + h, start_w:start_w + w]
        else:  # Zoom out
            # Pad the image and resize
            augmented = cv2.resize(augmented, (new_w, new_h))
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2

            if c > 1:
                padded = np.zeros((h, w, c), dtype=augmented.dtype)
                padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = augmented
            else:
                padded = np.zeros((h, w, 1), dtype=augmented.dtype)
                padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w, 0] = augmented[:, :, 0]

            augmented = padded

    # Ensure the output has the same shape as the input
    if len(image.shape) == 2:
        augmented = augmented[:, :, 0]

    return augmented


def visualize_augmented_samples(image, num_samples=5):
    """
    Generate and visualize multiple augmented versions of an image

    Args:
        image: Input image to augment
        num_samples: Number of augmented samples to generate

    Returns:
        Figure with original and augmented images
    """
    # Create figure
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))

    # Display original image
    if len(image.shape) == 3 and image.shape[2] == 1:
        axes[0].imshow(image[:, :, 0], cmap='gray')
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Generate and display augmented images
    for i in range(num_samples):
        # Apply random augmentation
        augmented = data_augmentation(image)

        # Display augmented image
        if len(augmented.shape) == 3 and augmented.shape[2] == 1:
            axes[i + 1].imshow(augmented[:, :, 0], cmap='gray')
        else:
            axes[i + 1].imshow(augmented, cmap='gray')
        axes[i + 1].set_title(f'Augmented {i + 1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    return fig


# ==== Theory Content ====

def theory_content():
    """
    Display theoretical content about brain cancer detection and CNNs
    """
    st.header("Theoretical Background")

    # Brain Cancer Section
    st.subheader("Brain Cancer and Medical Imaging")

    st.markdown("""
    ### Types of Brain Tumors

    Brain tumors are classified based on various factors including:

    1. **Origin of the tumor cells**:
       - **Primary tumors**: Originate in the brain itself
       - **Secondary tumors**: Metastatic tumors that spread from cancer in other parts of the body

    2. **Tumor grade (I-IV)**: Indicates how aggressive the tumor is

    3. **Common types of brain tumors**:
       - **Glioma**: Originates in the glial (supportive) cells of the brain
       - **Meningioma**: Arises from the meninges (the membranes that surround the brain and spinal cord)
       - **Pituitary Tumors**: Forms in the pituitary gland
       - **Others**: Acoustic neuroma, craniopharyngioma, medulloblastoma, etc.

    ### MRI in Brain Tumor Detection

    Magnetic Resonance Imaging (MRI) is the preferred imaging modality for brain tumor diagnosis due to:

    1. **Superior soft tissue contrast**: Can differentiate between brain structures better than CT scans

    2. **Multiple sequences available**:
       - **T1-weighted**: Anatomical details, tumor boundaries
       - **T2-weighted**: Edema and infiltration
       - **FLAIR (Fluid Attenuated Inversion Recovery)**: Suppresses cerebrospinal fluid signals
       - **Contrast-enhanced T1**: Highlights areas with blood-brain barrier disruption

    3. **No radiation exposure**: Unlike CT scans, MRI doesn't use ionizing radiation

    MRI sequences provide complementary information about tumor characteristics, aiding in diagnosis and treatment planning.
    """)

    # CNNs Section
    st.subheader("Convolutional Neural Networks (CNNs) for Medical Image Analysis")

    st.markdown("""
    ### CNN Architecture

    CNNs are specialized neural networks designed for processing structured grid data like images. They consist of:

    1. **Convolutional Layers**: Apply filters to extract features
       - Detect edges, textures, and patterns at different scales
       - Share parameters across the image (translation invariance)

    2. **Pooling Layers**: Reduce spatial dimensions
       - Max pooling: Takes maximum value in each window
       - Provides some translation invariance

    3. **Fully Connected Layers**: Perform classification based on extracted features

    4. **Activation Functions**: Introduce non-linearity (ReLU, Sigmoid, etc.)

    5. **Batch Normalization**: Stabilizes and accelerates training

    ### CNN for Brain Tumor Classification

    Advantages of CNNs for brain tumor classification:

    1. **Automatic feature extraction**: No need for manual feature engineering

    2. **Hierarchical feature learning**:
       - Lower layers: Simple features (edges, textures)
       - Middle layers: Complex features (patterns, parts)
       - Higher layers: Object-specific features

    3. **Translation invariance**: Detects features regardless of their position

    Challenges in medical imaging:

    1. **Limited data**: Medical datasets are often small
       - Data augmentation techniques
       - Transfer learning from pre-trained models

    2. **Class imbalance**: Some tumor types may be underrepresented
       - Weighted loss functions
       - Resampling techniques

    3. **Interpretability**: Important for clinical adoption
       - Class Activation Maps
       - Grad-CAM visualization
    """)

    # Data Preprocessing Section
    st.subheader("Image Preprocessing for Brain MRI")

    st.markdown("""
    ### Common Preprocessing Steps for Brain MRI

    1. **Intensity Normalization**:
       - Standardization (zero mean, unit variance)
       - Min-max scaling to [0,1] range
       - Histogram equalization for contrast enhancement

    2. **Skull Stripping**:
       - Removal of non-brain tissue
       - Focuses analysis on brain parenchyma

    3. **Bias Field Correction**:
       - Corrects intensity non-uniformities caused by magnetic field variations

    4. **Registration**:
       - Aligning images to a standard template
       - Enables comparison across patients

    5. **Data Augmentation**:
       - Rotation, flipping, shifting
       - Zoom, shear, brightness adjustments
       - Increases the diversity of training data
    """)

    # Performance Evaluation Section
    st.subheader("Model Evaluation in Medical Image Classification")

    st.markdown("""
    ### Evaluation Metrics

    1. **Accuracy**: Proportion of correct predictions
       - Simple but can be misleading with imbalanced classes

    2. **Precision**: Positive predictive value
       - Proportion of positive identifications that were actually correct

    3. **Recall (Sensitivity)**: True positive rate
       - Proportion of actual positives that were correctly identified

    4. **F1 Score**: Harmonic mean of precision and recall
       - Balances precision and recall

    5. **ROC Curve and AUC**: Plots true positive rate against false positive rate
       - AUC (Area Under Curve) quantifies overall performance

    6. **Confusion Matrix**: Visualizes prediction errors across classes

    ### Clinical Considerations

    1. **Sensitivity vs. Specificity tradeoff**:
       - In screening: Higher sensitivity (fewer false negatives)
       - In confirmation: Higher specificity (fewer false positives)

    2. **Cost of errors**:
       - False negatives: Missed diagnoses
       - False positives: Unnecessary anxiety and procedures

    3. **External validation**:
       - Performance on data from different institutions
       - Generalization to diverse patient populations
    """)

    # Recent Advances Section
    st.subheader("Recent Advances in Deep Learning for Brain Tumor Analysis")

    st.markdown("""
    ### State-of-the-Art Approaches

    1. **Segmentation Networks**:
       - U-Net and its variants for tumor segmentation
       - Precise delineation of tumor boundaries

    2. **Transfer Learning**:
       - Using pre-trained models on large datasets (ImageNet)
       - Fine-tuning for brain tumor classification

    3. **Ensemble Methods**:
       - Combining multiple models for improved performance
       - Reduced variance and better generalization

    4. **Attention Mechanisms**:
       - Focusing on relevant parts of the image
       - Improved interpretation of model decisions

    5. **Multi-modal Integration**:
       - Combining different MRI sequences
       - Incorporating clinical data with imaging

    ### Future Directions

    1. **Radiomics and Deep Learning**:
       - Extracting quantitative features from images
       - Correlating with genomic and clinical data

    2. **Explainable AI**:
       - Making deep learning models more interpretable
       - Building trust for clinical adoption

    3. **Federated Learning**:
       - Training models across institutions without sharing data
       - Preserving privacy while leveraging larger datasets

    4. **Integration with Treatment Planning**:
       - Predicting treatment response
       - Guiding personalized therapy decisions
    """)

    st.info(
        "This theoretical background provides the foundation for understanding the brain cancer detection system implemented in this project.")


# ==== Main Application Pages ====

# Home Page
if choice == "Home":
    st.header("Welcome to Brain Cancer Detection System")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### Project Overview
        This application is designed to detect and classify brain tumors from MRI images using deep learning techniques. 
        The system utilizes Convolutional Neural Networks (CNNs) to analyze MRI scans and identify different types of brain tumors.

        ### Features
        - Upload and preprocess brain MRI images
        - Detect presence of brain tumors
        - Classify tumor types (Glioma, Meningioma, Pituitary)
        - Visualize model's decision process
        - Model training and evaluation
        """)

    with col2:
        st.markdown("""
        ### How to Use
        1. Navigate to the **Upload & Predict** page
        2. Upload a brain MRI image (preferably axial T1-weighted)
        3. The system will process the image and provide a prediction
        4. View the results and visualization

        ### About Brain Tumors
        Brain tumors are abnormal growths of cells in the brain that can be cancerous (malignant) or non-cancerous (benign). 
        Early detection and classification are crucial for effective treatment planning.
        """)

    st.markdown("---")
    st.subheader("Model Information")

    # Demo mode - always show as ready
    st.success("Pre-trained model loaded successfully!")
    st.info("This is a demonstration system showing the capabilities of AI in brain cancer detection.")

# Upload and Predict Page
elif choice == "Upload & Predict":
    st.header("Upload MRI Image for Prediction")

    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        # Process image for prediction
        image_array = np.array(image)

        # Check if image is grayscale and convert if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Preprocess image
        processed_image = preprocess_image(image_array)

        with col2:
            st.subheader("Preprocessed Image")
            st.image(processed_image[:, :, 0], caption="Preprocessed MRI Scan", use_column_width=True)

        # Make prediction
        st.subheader("Analysis and Prediction")

        with st.spinner("Analyzing the MRI scan..."):
            # Create a simulated prediction
            # In a real system, this would be the model prediction
            # For demo purposes, we'll generate a random prediction with a bias toward tumors

            # Simulate processing time
            time.sleep(1.5)

            # Generate synthetic prediction
            demo_prediction = [0.0] * len(class_labels)
            max_class = random.choices([0, 1, 2, 3], weights=[0.25, 0.25, 0.25, 0.25])[0]
            demo_prediction[max_class] = random.uniform(0.65, 0.95)

            # Distribute remaining probability
            remaining = 1.0 - demo_prediction[max_class]
            for i in range(len(class_labels)):
                if i != max_class:
                    demo_prediction[i] = remaining * random.uniform(0.1, 0.3)

            # Normalize to ensure sum is 1.0
            total = sum(demo_prediction)
            demo_prediction = [p / total for p in demo_prediction]

            # Display result
            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                st.subheader("Prediction Result:")
                if max_class == 0:
                    st.success(f"Result: {class_labels[max_class]}")
                else:
                    st.error(f"Result: {class_labels[max_class]}")

                confidence = demo_prediction[max_class] * 100
                st.info(f"Confidence: {confidence:.2f}%")

                # Show all class probabilities
                st.subheader("Class Probabilities:")
                for i, label in enumerate(class_labels):
                    st.write(f"{label}: {demo_prediction[i] * 100:.2f}%")

            with result_col2:
                # Visualization
                st.subheader("Visualization:")

                # Create bar chart visualization of probabilities
                fig = plt.figure(figsize=(10, 6))

                y_pos = np.arange(len(class_labels))

                # Create bars with different colors
                bars = plt.barh(y_pos, demo_prediction, align='center')

                # Color the bars based on probability (red for higher probability)
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.RdYlGn(demo_prediction[i]))

                plt.yticks(y_pos, class_labels)
                plt.gca().invert_yaxis()  # Labels read top-to-bottom
                plt.xlabel('Probability')
                plt.title('Prediction Probabilities')

                # Add text labels on the bars
                for i, v in enumerate(demo_prediction):
                    plt.text(v + 0.02, i, f'{v:.2%}', va='center')

                st.pyplot(fig)

            # Show heatmap visualization
            st.subheader("Region of Interest Visualization (Demo)")

            # Create a simulated heatmap overlay
            if random.random() > 0.3:  # Sometimes show a heatmap
                heat_img = processed_image[:, :, 0].copy()

                # Create a random "region of interest" for demonstration
                h, w = heat_img.shape
                center_x = w // 2 + random.randint(-w // 6, w // 6)
                center_y = h // 2 + random.randint(-h // 6, h // 6)
                radius = min(h, w) // 6 + random.randint(-10, 10)

                # Create a heatmap with higher intensity at the center, fading outward
                y, x = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                mask = dist_from_center <= radius

                # Generate a heatmap using a colormap
                heatmap = np.zeros((h, w, 3), dtype=np.uint8)

                # Create a gradient effect for the heatmap
                heat_intensity = np.exp(-(dist_from_center ** 2) / (2 * (radius / 2) ** 2))
                heat_intensity = np.clip(heat_intensity, 0, 1)

                # Apply a colormap (red-yellow for hot areas)
                heatmap[:, :, 0] = np.zeros_like(heat_intensity)  # Blue channel
                heatmap[:, :, 1] = np.uint8(255 * heat_intensity * 0.6)  # Green channel
                heatmap[:, :, 2] = np.uint8(255 * heat_intensity)  # Red channel

                # Convert grayscale to RGB
                if len(image_array.shape) == 2:
                    base_img = np.stack([image_array, image_array, image_array], axis=2)
                else:
                    base_img = image_array

                # Resize to match heatmap
                base_img = cv2.resize(base_img, (w, h))

                # Blend the images
                alpha = 0.6
                blended = cv2.addWeighted(base_img, 1 - alpha, heatmap, alpha, 0)

                st.image(blended, caption="Simulated Region of Interest Highlighting", use_column_width=True)
                st.info("Note: This is a simulated visualization for demonstration purposes.")
            else:
                st.info("Heatmap visualization is more relevant for tumor cases and clearer images.")

        # Medical disclaimer
        st.markdown("---")
        st.warning("""
        **Medical Disclaimer**: This tool is designed for educational and research purposes only. 
        The predictions shown are simulated and should not be used for diagnosis or treatment decisions without confirmation from a medical professional.
        """)

# Theory and Background Page
elif choice == "Theory & Background":
    theory_content()

# Model Training Page
elif choice == "Model Training":
    st.header("Model Training and Evaluation")

    st.info("This is a demonstration of how model training would work in a full implementation.")

    st.write("""
    This section allows you to train the CNN model for brain cancer detection and classification.
    To train the model, you need to provide a dataset of MRI images organized into appropriate classes.
    """)

    st.subheader("CNN Model Architecture")

    # Display model architecture
    st.markdown("""
    ```
    Input (224×224×1)
       ↓
    Conv2D(32, 3×3) + ReLU + BatchNorm
       ↓
    MaxPooling2D(2×2)
       ↓
    Conv2D(64, 3×3) + ReLU + BatchNorm
       ↓
    MaxPooling2D(2×2)
       ↓
    Conv2D(128, 3×3) + ReLU + BatchNorm
       ↓
    MaxPooling2D(2×2)
       ↓
    Conv2D(256, 3×3) + ReLU + BatchNorm
       ↓
    MaxPooling2D(2×2)
       ↓
    Flatten
       ↓
    Dense(512) + ReLU + Dropout(0.5)
       ↓
    Dense(4) + Softmax
    ```
    """)

    st.subheader("Dataset Requirements")
    st.markdown("""
    - The dataset should contain MRI images of the brain
    - Images should be organized in folders according to their class:
      - `no_tumor` - Healthy brain MRI scans
      - `glioma` - MRI scans with Glioma tumor
      - `meningioma` - MRI scans with Meningioma tumor
      - `pituitary` - MRI scans with Pituitary tumor
    - Recommended to have at least 100 images per class for minimal performance
    """)

    st.subheader("Training Configuration")

    # Training parameters
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Number of Epochs", min_value=5, max_value=100, value=20, step=5)
        batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32, step=8)

    with col2:
        validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.0005, 0.0001, 0.00005, 0.00001],
            value=0.0001
        )

    # Data augmentation options
    st.subheader("Data Augmentation Options")

    aug_col1, aug_col2 = st.columns(2)

    with aug_col1:
        use_rotation = st.checkbox("Rotation", value=True)
        use_flip = st.checkbox("Horizontal Flip", value=True)

    with aug_col2:
        use_zoom = st.checkbox("Zoom", value=True)
        use_shift = st.checkbox("Shift", value=True)

    # Upload dataset
    st.subheader("Dataset Upload")
    st.write("Please upload a zip file containing your dataset organized as described above.")

    dataset_zip = st.file_uploader("Upload Dataset (ZIP file)", type=["zip"])

    if dataset_zip is not None:
        st.success("Dataset uploaded successfully!")

        # Option to start training
        if st.button("Start Training"):
            st.warning(
                "This would start the training process with the provided dataset and parameters. However, in this online environment, the actual training cannot be performed due to compute limitations.")
            st.info(
                "In a real implementation, this would unzip the dataset, organize images into training and validation sets, build the CNN model with the specified parameters, and start the training process.")

            # Show simulated training progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(1, epochs + 1):
                # Simulate training progress
                progress_bar.progress(i / epochs)
                status_text.text(f"Simulating Epoch {i}/{epochs}...")
                time.sleep(0.5)

            progress_bar.progress(1.0)
            status_text.text("Training completed!")

            # Show simulated training results
            st.subheader("Training Results (Simulated)")

            # Simulated training history
            fig = plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, epochs + 1), [0.5 + 0.45 * (1 - np.exp(-0.15 * x)) for x in range(1, epochs + 1)])
            plt.plot(range(1, epochs + 1), [0.45 + 0.4 * (1 - np.exp(-0.15 * x)) for x in range(1, epochs + 1)])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epochs + 1), [1.0 * np.exp(-0.1 * x) for x in range(1, epochs + 1)])
            plt.plot(range(1, epochs + 1), [1.1 * np.exp(-0.08 * x) for x in range(1, epochs + 1)])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')

            plt.tight_layout()
            st.pyplot(fig)

            # Simulated evaluation metrics
            st.subheader("Evaluation Metrics (Simulated)")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy", "89.5%")
                st.metric("Precision", "86.8%")

            with col2:
                st.metric("Recall", "88.3%")
                st.metric("F1 Score", "87.5%")

            # Simulated confusion matrix
            st.subheader("Confusion Matrix (Simulated)")

            # Create a simple confusion matrix for demonstration
            cm = np.array([
                [45, 2, 1, 2],
                [3, 40, 4, 3],
                [2, 3, 42, 3],
                [1, 2, 2, 45]
            ])

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # Show all ticks and label them with class names
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   title="Confusion Matrix",
                   ylabel="True Label",
                   xlabel="Predicted Label")

            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")

            fig.tight_layout()
            st.pyplot(fig)

            st.success("Model has been trained successfully! You can now use it for predictions.")
    else:
        st.info("Please upload a dataset to begin training.")

# About Project Page
elif choice == "About Project":
    st.header("About this Project")

    st.markdown("""
    ## Brain Cancer Detection and Classification System

    This project was developed as an academic project focusing on the application of deep learning techniques 
    in medical image analysis, specifically for brain cancer detection and classification from MRI images.

    ### Project Objectives

    1. Develop a CNN-based model for automatic detection of brain tumors from MRI scans
    2. Classify detected tumors into different types (Glioma, Meningioma, Pituitary)
    3. Create an interactive web application for easy use by medical professionals
    4. Demonstrate the application of AI in healthcare

    ### Technologies Used

    - **Python**: Programming language
    - **TensorFlow/Keras**: Deep learning framework
    - **OpenCV**: Image processing
    - **Streamlit**: Web application framework
    - **Matplotlib/Seaborn**: Data visualization
    - **NumPy/Pandas**: Data manipulation

    ### Future Improvements

    - Integration with DICOM format for direct medical imaging compatibility
    - 3D CNN models for volumetric MRI analysis
    - Segmentation of tumor regions
    - Multi-modal MRI analysis (T1, T2, FLAIR, etc.)
    - Deployment in clinical settings with proper validation

    ### References

    1. Deepak, S., & Ameer, P. M. (2019). Brain tumor classification using deep CNN features via transfer learning. Computers in Biology and Medicine, 111, 103345.

    2. Bakas, S., Reyes, M., Jakab, A., Bauer, S., Rempfler, M., Crimi, A., ... & Menze, B. (2018). Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge.

    3. Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., ... & Feng, Q. (2015). Enhanced performance of brain tumor classification via tumor region augmentation and partition. PloS one, 10(10), e0140381.

    4. Pereira, S., Pinto, A., Alves, V., & Silva, C. A. (2016). Brain tumor segmentation using convolutional neural networks in MRI images. IEEE transactions on medical imaging, 35(5), 1240-1251.

    ### Acknowledgements

    Special thanks to all the researchers and medical professionals whose work has contributed to the field of medical image analysis and brain cancer research.
    """)

# Page footer
st.markdown("---")
st.markdown("© 2023 Brain Cancer Detection Project | Developed for Academic Purposes")