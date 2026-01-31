import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_xray_model(num_classes=14,input_shape=(224,224,3)):

    base_model=applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable=False

    inputs=layers.Input(shape=input_shape)
    x=base_model(inputs,training=False) 
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(1024,activation='relu')(x)
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(num_classes,activation='sigmoid')(x)
    model=models.Model(inputs,outputs,name='chest_xray_model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=["AUC", 'accuracy']
    )

    return model

if __name__ == "__main__":
    model=build_xray_model()
    model.summary()
    print("\nMODEL BUILT SUCCESSFULLY\n")