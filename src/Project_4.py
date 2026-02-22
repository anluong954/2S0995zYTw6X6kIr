import numpy as np
import pandas as pd
import tensorflow as tf
import os

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# neural networks
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# pre-trained models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50  # ResNet (50 layers)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2  # MobileNet (version 2)
from tensorflow.keras.applications.efficientnet import EfficientNetB5  # EfficientNet

def accuracy_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def f1_score_macro_np(y_true, y_pred, num_classes: int | None = None) -> float:
    """
    Macro-F1 for single-label multiclass classification.
    Uses per-class F1 (with 0 when precision+recall==0), then averages across classes.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size == 0:
        return 0.0

    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)

    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        denom = (2 * tp + fp + fn)
        f1_c = (2 * tp / denom) if denom != 0 else 0.0
        f1s.append(f1_c)

    return float(np.mean(f1s))

# Data Exploration
train_data_dir = 'C:/Users/anluo/OneDrive/Desktop/Projects/Project 4/images/training'
test_data_dir = 'C:/Users/anluo/OneDrive/Desktop/Projects/Project 4/images/testing'
img_height, img_width, img_chns = 180, 180, 3
batch_size = 32

train_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

classes = train_imgs.class_names
num_classes = len(classes)
print(classes)

plt.figure(figsize=(10, 10))
for images, labels in train_imgs.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes[int(labels[i])])
        plt.axis("off")

plt.show()

def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, num_classes: int) -> dict:
    y_true = []
    y_pred = []

    for batch_images, batch_labels in dataset:
        probs = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(probs, axis=1)

        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    acc = accuracy_score_np(y_true, y_pred)
    f1 = f1_score_macro_np(y_true, y_pred, num_classes=num_classes)

    return {"accuracy": float(acc), "f1_score": float(f1)}

# Modeling

# VGG-16
def vgg_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    vgg16 = VGG16(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in vgg16.layers:
        layer.trainable = False

    x = vgg16.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    vgg_model = Model(inputs=vgg16.input, outputs=output_layer)
    vgg_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_vgg16 = vgg_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return vgg_model, trained_vgg16


# Training
vgg_model, trained_vgg16 = vgg_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig2 = plt.gcf()
plt.plot(trained_vgg16.history["accuracy"], label="accuracy")
plt.plot(trained_vgg16.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

# Saving
sizes = {}

# Save under a predictable folder next to this script (more portable than hardcoding)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

vgg_save_path = os.path.join(save_dir, "vgg16.keras")
vgg_model.save(vgg_save_path)

sizes["vgg"] = os.path.getsize(vgg_save_path) / 1e6
print(f"VGG16 model size: {sizes['vgg']:.2f} MB")

vgg_model_evaluation = evaluate_model(vgg_model, test_imgs, num_classes=num_classes)

# Resnet
def resnet_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    resnet = ResNet50(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in resnet.layers:
        layer.trainable = False

    x = resnet.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    resnet_model = Model(inputs=resnet.input, outputs=output_layer)
    resnet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_resnet = resnet_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return resnet_model, trained_resnet


# Training
resnet_model, trained_resnet = resnet_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig3 = plt.gcf()
plt.plot(trained_resnet.history["accuracy"], label="accuracy")
plt.plot(trained_resnet.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

resnet_model_evaluation = evaluate_model(resnet_model, test_imgs, num_classes=num_classes)

# MobileNet
def mobilenet_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    mobilenet = MobileNetV2(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in mobilenet.layers:
        layer.trainable = False

    x = mobilenet.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    mobilenet_model = Model(inputs=mobilenet.input, outputs=output_layer)
    mobilenet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_mobilenet = mobilenet_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return mobilenet_model, trained_mobilenet


# Training
mobilenet_model, trained_mobilenet = mobilenet_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig4 = plt.gcf()
plt.plot(trained_mobilenet.history["accuracy"], label="accuracy")
plt.plot(trained_mobilenet.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

mobilenet_model_evaluation = evaluate_model(mobilenet_model, test_imgs, num_classes=num_classes)

# EfficientNet
def effnet_transfer(train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes):
    num_classes = len(classes)

    effnet = EfficientNetB5(
        weights='imagenet',
        input_shape=(img_height, img_width, img_chns),
        include_top=False,
        pooling='avg'  # outputs a vector already (GlobalAveragePooling2D)
    )

    for layer in effnet.layers:
        layer.trainable = False

    x = effnet.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    effnet_model = Model(inputs=effnet.input, outputs=output_layer)
    effnet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trained_effnet = effnet_model.fit(
        train_imgs,
        epochs=10,
        validation_data=val_imgs
    )
    return effnet_model, trained_effnet


# Training
effnet_model, trained_effnet = effnet_transfer(
    train_imgs, val_imgs, test_imgs, img_height, img_width, img_chns, classes
)

# Evaluation
fig5 = plt.gcf()
plt.plot(trained_effnet.history["accuracy"], label="accuracy")
plt.plot(trained_effnet.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

effnet_model_evaluation = evaluate_model(effnet_model, test_imgs, num_classes=num_classes)

# Conclusion

values = {
    'accuracy': [
        vgg_model_evaluation['accuracy'],
        resnet_model_evaluation['accuracy'],
        mobilenet_model_evaluation['accuracy'],
        effnet_model_evaluation['accuracy'],
    ],
    'f1_score': [
        vgg_model_evaluation['f1_score'],
        resnet_model_evaluation['f1_score'],
        mobilenet_model_evaluation['f1_score'],
        effnet_model_evaluation['f1_score'],
    ],
}

df = pd.DataFrame(values, index = ['vgg', 'resnet', 'mobilenet', 'efficientnet'])
print(df)