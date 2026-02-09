import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import get_train_val_generators

def diagnose(img_path):

    model = load_model('checkpoints/xray_model_best.keras')
    _, _, all_labels = get_train_val_generators()
    

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

  
    preds = model.predict(img_array)
    top_idx = np.argmax(preds[0])
    print(f"\n--- PREDICTION DEBUG ---")
    print(f"Top Prediction: '{all_labels[top_idx]}' with Probability: {preds[0][top_idx]:.5f}")
    print(f"Raw Probabilities (First 5): {preds[0][:5]}")


    backbone = model.layers[1] 
    output_layer = model.layers[5] 
    
    with tf.GradientTape() as tape:
        conv_output = backbone(img_array)
        tape.watch(conv_output)
        
        x = model.layers[2](conv_output) # Pooling
        x = model.layers[3](x)           # Dense
        preds_manual = output_layer(x)   # Output
        
        target_score = preds_manual[:, top_idx]
    
    grads = tape.gradient(target_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    print(f"\nGRADIENT DEBUG ")
    print(f"Max Gradient Value: {tf.reduce_max(grads):.8f}")
    print(f"Mean Gradient Value: {tf.reduce_mean(grads):.8f}")
    
    if tf.reduce_max(grads) == 0:
        print("CRITICAL: Gradients are ZERO. The model learned nothing or 'Dead ReLU' problem.")
        return

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    print(f"\nHEATMAP DEBUG")
    print(f"Heatmap Max Value (Before Norm): {tf.reduce_max(heatmap):.5f}")
    
    if tf.reduce_max(heatmap) == 0:
        print("CRITICAL: Heatmap is all zeros. (ReLU clipped everything).")

diagnose("data/images/00000013_026.png")
