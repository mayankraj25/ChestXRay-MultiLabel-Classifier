import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generates the Grad-CAM heatmap by manually stepping through the model layers.
    Robust to Nested Models (ResNet inside a Functional Model).
    """
    # ---------------------------------------------------------
    # 1. DECONSTRUCT THE MODEL
    # ---------------------------------------------------------
    # We know the architecture from model.py:
    # Layer [0]: Input
    # Layer [1]: ResNet50 (The Backbone) -> Outputs 7x7x2048 Feature Map
    # Layer [2]: GlobalAveragePooling2D
    # Layer [3]: Dense (1024)
    # Layer [4]: Dropout
    # Layer [5]: Dense (Output)
    
    # Isolate the Feature Extractor (ResNet)
    backbone = model.layers[1] 
    
    # Isolate the Classifier Head (The rest of the layers)
    # We'll run these manually later
    pooling_layer = model.layers[2]
    dense_1 = model.layers[3]
    # We skip Dropout (layer 4) during inference/GradCAM
    output_layer = model.layers[5]

    # ---------------------------------------------------------
    # 2. RECORD GRADIENTS
    # ---------------------------------------------------------
    with tf.GradientTape() as tape:
        # A. Get the Feature Map from the Backbone
        # (This is the 7x7x2048 volume)
        conv_output = backbone(img_array)
        
        # We need to watch this tensor to calculate gradients relative to it
        tape.watch(conv_output)
        
        # B. Pass through the Classifier Head manually
        x = pooling_layer(conv_output)
        x = dense_1(x)
        # Skip dropout
        preds = output_layer(x)
        
        # C. Select the Class to Explain
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        # Get the score for the target class
        class_channel = preds[:, pred_index]

    # ---------------------------------------------------------
    # 3. COMPUTE HEATMAP
    # ---------------------------------------------------------
    # Calculate gradient of the Class Score w.r.t. the Feature Map
    grads = tape.gradient(class_channel, conv_output)

    # Global Average Pooling of Gradients (Importance Weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply Feature Map by Weights
    # We work with the first image in the batch [0]
    conv_output = conv_output[0]
    
    # Matrix multiplication: (7x7x2048) @ (2048) -> (7x7)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (Keep only positive influence) & Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    """

    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # 🔹 Resize heatmap from 7x7 → 224x224
    heatmap = cv2.resize(heatmap, (224, 224))

    # 🔹 Safe normalization
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # Convert to 0–255
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, jet, alpha, 0)

    # Convert BGR → RGB for matplotlib
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original X-Ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load your best model
    model = load_model('checkpoints/xray_model_finetuned.keras')
    
    # 2. Pick a test image
    # Ensure this file exists!
    test_img_path = "/Users/mayankraj/Desktop/My_Projects/ChestXray_Project/data/images/00000096_006.png" 
    
    # 3. Preprocess
    img = tf.keras.preprocessing.image.load_img(test_img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Make batch of 1
    img_array /= 255.0 # Normalize

    # 4. Generate & Show
    # Note: We don't pass layer names anymore. The function handles it.
    heatmap = make_gradcam_heatmap(img_array, model)
    save_and_display_gradcam(test_img_path, heatmap)