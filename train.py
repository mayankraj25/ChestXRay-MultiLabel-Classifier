import tensorflow as tf
from data_loader import get_train_val_generators
from model import build_xray_model
import os

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

train_gen, val_gen, labels = get_train_val_generators()
model=build_xray_model(num_classes=len(labels))

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/xray_model_best.keras',
    save_best_only=True,
    monitor='val_auc',     
    mode='max',           
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.1,            
    patience=2,            
    mode='max',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5,            
    mode='max',
    restore_best_weights=True
)

print("\n STARTING TRAINING...\n")

import time
start_time = time.time()

history=model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[checkpoint, reduce_lr, early_stop],
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size
)

end_time = time.time()
total_time = end_time - start_time

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print(f"\nTotal training time: {hours}h {minutes}m {seconds}s")
print("\n TRAINING COMPLETED\n")