# -*- coding: utf-8 -*-
"""
AER850 - Project 2
Step 5: Testing Model 1 and Model 2 on three test images
Shows TRUE label and PREDICTED label on each figure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -----------------------------------------------------
# Basic settings (
# -----------------------------------------------------
IMG_HEIGHT, IMG_WIDTH = 500, 500

# class mapping 

class_indices = {'crack': 0, 'missing-head': 1, 'paint-off': 2}
idx_to_class = {v: k for k, v in class_indices.items()}
print("Class mapping:", idx_to_class)

# -----------------------------------------------------
# Helper: load + preprocess
# -----------------------------------------------------
def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# -----------------------------------------------------
# Test image paths (one per class)
# -----------------------------------------------------
base_dir = os.path.join(os.getcwd(), "Data")
test_dir = os.path.join(base_dir, "test")

test_images = [
    ("crack",        os.path.join(test_dir, "crack",        "test_crack.jpg")),
    ("missing-head", os.path.join(test_dir, "missing-head", "test_missinghead.jpg")),
    ("paint-off",    os.path.join(test_dir, "paint-off",    "test_paintoff.jpg")),
]

print("\nChecking test image paths:")
for true_label, p in test_images:
    print(f"{true_label:12s} -> {os.path.isfile(p)} | {p}")

# -----------------------------------------------------
# Test one model and save 3 figures
# -----------------------------------------------------
def test_model_and_save_pics(model_file):
    print("\n===============================")
    print(f"Testing model: {model_file}")
    print("===============================")

    model = tf.keras.models.load_model(model_file)
    print("Model loaded!")

    model_name = os.path.splitext(model_file)[0] 

    for true_label, img_path in test_images:
        # load & predict
        orig_img, img_batch = load_and_preprocess_image(img_path)
        preds = model.predict(img_batch, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_class = idx_to_class[pred_idx]
        confidence = preds[pred_idx]

        print(
            f"{os.path.basename(img_path)} | TRUE: {true_label:12s} "
            f"| PRED: {pred_class:12s} (conf = {confidence:.3f})"
        )

        # make single-image figure
        plt.figure(figsize=(4, 4))
        plt.imshow(orig_img)
        plt.axis("off")
        # 🔹 show BOTH true and pred here
        plt.title(
            f"True: {true_label}\nPred: {pred_class} (Conf: {confidence:.2f})",
            fontsize=11
        )

        
        out_name = f"{model_name}_{true_label}.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Saved:", out_name)

# -----------------------------------------------------
# Run tests for both models
# -----------------------------------------------------
for model_file in ["model1_best.keras", "model2_best.keras"]:
    if os.path.exists(model_file):
        test_model_and_save_pics(model_file)
    else:
        print(f" Model not found: {model_file}")


