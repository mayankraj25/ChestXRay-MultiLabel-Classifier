import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_val_generators(csv_path='data/clean_labels.csv', img_dir='data/images'):
    df = pd.read_csv(csv_path)

    all_labels=[x for x in df.columns if x not in ['Image Index', 'Finding Labels', 'path', 'No Finding','Follow-up #','Patient Age','Patient Gender','View Position','OriginalImageWidth','OriginalImageHeight','OriginalImagePixelSpacing_x','OriginalImagePixelSpacing_y',]]
    print(f"Detecting {len(all_labels)} pathologies: {all_labels}")

    train_df, val_df=train_test_split(df, test_size=0.2,random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen=ImageDataGenerator(rescale=1./255)

    IMG_SIZE=(224,224)
    BATCH_SIZE=32

    train_gen=train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col=all_labels,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw',
        shuffle=True
    )

    val_gen=val_datagen.flow_from_dataframe(
        val_df,
        x_col='path',
        y_col=all_labels,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw',
        shuffle=False
    )

    return train_gen, val_gen, all_labels

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    train_gen, val_gen, all_labels = get_train_val_generators()
    x_batch, y_batch = next(train_gen)

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(x_batch[i])
        labels_present = [all_labels[j] for j in range(len(all_labels)) if y_batch[i][j] == 1]
        title = ", ".join(labels_present) if labels_present else "Healthy"
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()