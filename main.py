import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import stats
from sklearn.metrics import confusion_matrix

# Read the dataset
skin_df = pd.read_csv('/Users/aaditya/Desktop/skin cancer/data/HAM10000_metadata.csv')
    
SIZE = 32

# Encoding labels
le = LabelEncoder()
le.fit(skin_df['dx'])
print("Classes: ", list(le.classes_))

skin_df['label'] = le.transform(skin_df["dx"])
print("\nSample of the dataset:")
print(skin_df.sample(10))

# Data distribution visualization
fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type')

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex')

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count', size=12)
ax3.set_title('Localization')

ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red')
ax4.set_title('Age')

plt.tight_layout()
plt.show()

# Balance the dataset
print("\nOriginal class distribution:")
print(skin_df['label'].value_counts())

# Separate classes and resample
n_samples = 500
df_balanced = pd.concat([
    resample(skin_df[skin_df['label'] == i], 
            replace=True, 
            n_samples=n_samples, 
            random_state=42) 
    for i in range(7)
])

print("\nBalanced class distribution:")
print(df_balanced['label'].value_counts())

# Image loading setup
BASE_DIR = '/Users/aaditya/Desktop/skin cancer/data/HAM10000_images_part_1'

image_path = {os.path.splitext(os.path.basename(x))[0]: x
             for x in glob(os.path.join(BASE_DIR, '*.jpg'))}

def load_image(path):
    try:
        if path is None:
            return None
        img = Image.open(path)
        return np.asarray(img.resize((SIZE, SIZE)))
    except Exception as e:
        print(f"Error loading image at {path}: {str(e)}")
        return None

# Update the image loading code
df_balanced['path'] = df_balanced['image_id'].map(image_path.get)

print("Number of images found:", len(image_path))
print("Sample image paths:", list(image_path.values())[:5])
print("Number of None paths:", df_balanced['path'].isna().sum())

df_balanced['image'] = df_balanced['path'].map(load_image)
df_balanced = df_balanced.dropna(subset=['image'])

# Display sample images
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize=(4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
plt.show()

# Prepare data for modeling
X = np.asarray(df_balanced['image'].tolist())
X = X/255.  # Normalize pixel values
Y = df_balanced['label']
Y_cat = tf.keras.utils.to_categorical(Y, num_classes=7)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# Train model
batch_size = 16 
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=2
)

# Plot training history - Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], 'y', label='Training loss')
plt.plot(history.history['val_loss'], 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training history - Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'y', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

# Prediction and Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5)
plt.title('Confusion Matrix')
plt.show()

# Plot incorrect predictions
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.figure(figsize=(8, 6))
plt.bar(np.arange(7), incorr_fraction)
plt.title('Fraction of Incorrect Predictions by Class')
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.show()

# Save the model
model.save('skin_cancer_model.h5')
print("\nModel saved as 'skin_cancer_model.h5'")