import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import get_train_val_generators
import os

# 1. SETUP
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# 2. GET DATA (Same as before)
train_gen, val_gen, labels = get_train_val_generators()

# 3. LOAD YOUR BEST MODEL
print("Loading the warmed-up model...")
model = load_model('checkpoints/xray_model_best.keras')

# 4. UNFREEZE THE BACKBONE
# We want to unfreeze the last block of ResNet50.
# ResNet50 has ~175 layers. We unfreeze the top ~30.
base_model = model.layers[1] # This is the 'resnet50' layer inside your model
base_model.trainable = True

# We need to freeze the early layers (keep generic shapes) 
# and ONLY unfreeze the final block (specific textures).
# For ResNet50, the distinct blocks usually end around layer 140.
fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Backbone layers total: {len(base_model.layers)}")
print(f"Frozen layers: {fine_tune_at}")
print(f"Trainable layers: {len(base_model.layers) - fine_tune_at}")

# 5. RE-COMPILE (CRITICAL STEP)
# We MUST recompile to apply the 'trainable' changes.
# We use a Very Low Learning Rate (1e-5) to nudge weights gently.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['AUC', 'binary_accuracy']
)

model.summary()

# 6. DEFINE CALLBACKS
# We save to a NEW file so we don't overwrite the old one if this goes wrong.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/xray_model_finetuned.keras',
    save_best_only=True,
    monitor='val_auc',
    mode='max',
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    mode='max',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=8,
    restore_best_weights=True
)

# 7. TRAIN (FINE-TUNE)
print("\n--- Starting Fine-Tuning ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,  # Another 20 epochs of focused training
    callbacks=[checkpoint, reduce_lr, early_stop],
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size
)

print("\n--- Fine-Tuning Complete ---")