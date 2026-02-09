import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import get_train_val_generators
import os

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
train_gen, val_gen, labels = get_train_val_generators()

print("Loading the warmed-up model...")
model = load_model('checkpoints/xray_model_best.keras')

base_model.trainable = True

fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Backbone layers total: {len(base_model.layers)}")
print(f"Frozen layers: {fine_tune_at}")
print(f"Trainable layers: {len(base_model.layers) - fine_tune_at}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['AUC', 'binary_accuracy']
)

model.summary()
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

print("\n--- Starting Fine-Tuning ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,  
    callbacks=[checkpoint, reduce_lr, early_stop],
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size
)

print("\nFine-Tuning Complete !")
