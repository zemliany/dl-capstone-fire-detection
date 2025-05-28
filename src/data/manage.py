# src/data/data_prep.py

import io
import sys
import base64
from datetime import datetime
import kagglehub
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


def init_dataset(dataset_name: str, destination_dir: str = 'dataset/fire_dataset'):
    """
        Initializes dataset: loads from Kaggle if it doesn't exist locally,
        and copies its content to the desired destination_dir.

        :param dataset_name: ID of the dataset on Kaggle (e.g. "phylake1337/fire-dataset").
        :param destination_dir: Path to the directory where the dataset should be copied.
                                This will contain 'fire_images' and 'non_fire_images' subfolders.
    """

    final_fire_images_path = os.path.join(destination_dir, 'fire_images')
    final_non_images_path = os.path.join(destination_dir, 'non_fire_images')

    if (os.path.exists(final_fire_images_path) and os.listdir(final_fire_images_path) and
        os.path.exists(final_non_images_path) and os.listdir(final_non_images_path)):
        print(f"Dataset '{dataset_name}' already exists in '{destination_dir}' with desired structure. Skipping download.")
        return

    try:
        print(f"Downloading dataset {dataset_name} from Kaggle...")
        downloaded_cache_path = kagglehub.dataset_download(dataset_name)

        print("Path to downloaded dataset cache:", downloaded_cache_path)
        print(f"Preparing dataset in: {destination_dir}...")

        if os.path.exists(destination_dir):
            print(f"Directory '{destination_dir}' exists. Cleaning it before copying...")
            shutil.rmtree(destination_dir)
        os.makedirs(destination_dir, exist_ok=True)
        
        actual_source_data_dir = os.path.join(downloaded_cache_path, 'fire_dataset')
        
        if not os.path.exists(actual_source_data_dir) and os.path.isdir(os.path.join(downloaded_cache_path, 'Fire_dataset')):
            actual_source_data_dir = os.path.join(downloaded_cache_path, 'Fire_dataset')

        if not os.path.exists(actual_source_data_dir) or not os.path.isdir(actual_source_data_dir):
            raise FileNotFoundError(f"Expected source directory '{actual_source_data_dir}' (or similar) not found after download. "
                                    f"Please check the structure within '{downloaded_cache_path}'.")

        print(f"Actual data source to copy from: {actual_source_data_dir}")

        print(f"Copying contents from '{actual_source_data_dir}' to '{destination_dir}'...")
        for item in os.listdir(actual_source_data_dir):
            s = os.path.join(actual_source_data_dir, item)
            d = os.path.join(destination_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        print(f"Dataset successfully downloaded and prepared in: {destination_dir}")

    except Exception as e:
        print(f"Error downloading or preparing dataset: {e}")
        print("Please ensure the dataset ID is correct and you have Kaggle API credentials configured.")
        print("Also check the expected structure of the downloaded dataset from Kaggle.")
        sys.exit(1)


def prepare_dataset_and_generators(
    dataset_base_dir: str = 'dataset',
    img_height: int = 224,
    img_width: int = 224,
    batch_size: int = 32,
    test_size: float = 0.15,
    validation_size: float = 0.15
):
    """
        Collects image paths, splits data, creates generators
        and generates HTML parts of the dataset training report.

        :param dataset_base_dir: Base directory where fire_images/ and non_fire_images/ folders are located.
        :param img_height: Image height for the generator.
        :param img_width: Image width for the generator.
        :param batch_size: Batch size for the generator.
        :param test_size: Data portion for the test set.
        :param validation_size: Data portion for the validation set.
        :return: (train_generator, validation_generator, test_generator, html_content_parts)
    """

    html_content_parts = []
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    try:
        sys.stdout = redirected_output # Redirect stdout to capture logs

        fire_dir = os.path.join(dataset_base_dir, 'fire_images')
        no_fire_dir = os.path.join(dataset_base_dir, 'non_fire_images')

        image_paths = []
        labels = []

        # fire images threaded as class 1
        if os.path.exists(fire_dir):
            for img_name in os.listdir(fire_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_paths.append(os.path.join(fire_dir, img_name))
                    labels.append(1) # 1 = fire
        else:
            print(f"Warn: Fodler '{fire_dir}' not found or path to dataset might be incorrect")

        # non-fire images threaded as class 0
        if os.path.exists(no_fire_dir):
            for img_name in os.listdir(no_fire_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_paths.append(os.path.join(no_fire_dir, img_name))
                    labels.append(0) # 0 = no fire
        else:
            print(f"Warn: Folder '{no_fire_dir}' not found or path to dataset might be incorrect")

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        if len(image_paths) == 0:
            raise FileNotFoundError(f"Images not found in dir '{fire_dir}' OR '{no_fire_dir}'")

        # --- Data Split ---
        total_val_test_size = test_size + validation_size
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            image_paths, labels, test_size=total_val_test_size, random_state=42, stratify=labels
        )

        test_size_ratio_from_val_test = test_size / total_val_test_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=test_size_ratio_from_val_test, random_state=42, stratify=y_val_test
        )


        # --- Data Split Info ---
        print("\n--- Data Split Info ---")
        print(f"Total Amount of Images: {len(image_paths)}")
        print(f"Amount of images for train: {len(X_train)}")
        print(f"Amount of images for val: {len(X_val)}")
        print(f"Amount of images for test: {len(X_test)}")

        print("\nExample of paths and labels for a training set (first 10):")
        for i in range(min(10, len(X_train))):
            print(f"  {X_train[i]} -> {y_train[i]}")

        train_df = pd.DataFrame({'filename': X_train, 'class': y_train.astype(str)})
        val_df = pd.DataFrame({'filename': X_val, 'class': y_val.astype(str)})
        test_df = pd.DataFrame({'filename': X_test, 'class': y_test.astype(str)})

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col='class',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )

        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='filename',
            y_col='class',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='filename',
            y_col='class',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        # --- ImageGenerator ---
        print("\n--- ImageGenerator ---")
        print(f"Found {train_generator.samples} images for train with {len(train_generator.class_indices)} classes.")
        print(f"Found {validation_generator.samples} images for val with {len(validation_generator.class_indices)} classes.")
        print(f"Found {test_generator.samples} images for test with {len(test_generator.class_indices)} classes.")

        class_mapping = {0: "No Fire", 1: "Fire"}
        final_class_names = {}
        for idx_str, original_label_int in train_generator.class_indices.items():
            final_class_names[original_label_int] = class_mapping.get(original_label_int, f"Unknown ({original_label_int})")

        print(f"Індекси класів: {train_generator.class_indices}")
        print(f"Людсько-зрозумілі назви класів: {final_class_names}")


        # --- Visualization ---
        print("\n--- Приклад батчу з навчального генератора ---")
        images, labels_batch = next(train_generator)

        print(f"Розмір батчу (зображення): {images.shape}")
        print(f"Розмір батчу (мітки): {labels_batch.shape}")
        print("\nМітки зображень з батчу:")
        print(labels_batch)
        print("\nМітки зображень з батчу (людсько-зрозумілі):")
        human_readable_labels = [final_class_names.get(int(label), 'Unknown') for label in labels_batch]
        print(human_readable_labels)

        plt.figure(figsize=(10, 10))
        for i in range(min(9, batch_size)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            class_label_int = int(labels_batch[i])
            plt.title(f"Клас: {final_class_names.get(class_label_int, 'Unknown')}")
            plt.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        image_block_html = f"""
        <button class="collapsible">Images example</button>
        <div class="content">
            <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                <img src='data:image/png;base64,{image_base64}' alt='Img examples' style='max-width:100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ddd;'>
            </div>
        </div>
        """
        html_content_parts.append(image_block_html)

    finally:
        sys.stdout = old_stdout # Restore stdout

    # Get all captured output
    all_output = redirected_output.getvalue()

    logs_block_html = f"""
    <button class="collapsible">Logs output from data preparation</button>
    <div class="content">
        <pre>{all_output}</pre>
    </div>
    """
    html_content_parts.append(logs_block_html)

    return train_generator, validation_generator, test_generator, html_content_parts


# This block will only run if dataset_prep.py is executed directly
if __name__ == "__main__":
    print("Run data_prep.py для for test...")

    kaggle_dataset_id = "phylake1337/fire-dataset"
    dataset_target_dir = 'dataset/fire_dataset' # Ensure this matches config

    init_dataset(kaggle_dataset_id, destination_dir=dataset_target_dir)

    train_gen, val_gen, test_gen, html_parts = prepare_dataset_and_generators(
        dataset_base_dir=dataset_target_dir,
        img_height=224,
        img_width=224,
        batch_size=32,
        test_size=0.15,
        validation_size=0.15
    )

    print("\nData generators successfully created:")
    print(f"Train Generator: {train_gen.samples} samples")
    print(f"Validation Generator: {val_gen.samples} samples")
    print(f"Test Generator: {test_gen.samples} samples")

    # Creating a simple report for demonstration
    test_report_filename = 'reports/test_dataset_preparation_report.html'
    os.makedirs(os.path.dirname(test_report_filename), exist_ok=True)

    # Minimal HTML structure to show dataset_prep.py's output
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Data Preparation Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #e9f0f6; color: #333; line-height: 1.6; }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 25px; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .collapsible {{ background-color: #3498db; color: white; cursor: pointer; padding: 15px 20px; width: 100%; border: none; text-align: left; outline: none; font-size: 1.1em; transition: 0.4s; margin-top: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .active, .collapsible:hover {{ background-color: #2980b9; }}
            .collapsible:after {{ content: '\\02795'; font-size: 0.8em; color: white; float: right; margin-left: 5px; }}
            .active:after {{ content: '\\2796'; }}
            .content {{ padding: 0 18px; max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; background-color: #ffffff; border-left: 1px solid #ddd; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; border-radius: 0 0 5px 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            .content pre {{ background-color: #f8f8f8; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; border: 1px solid #eee; }}
            .content img {{ max-width: 100%; height: auto; display: block; margin: 15px auto; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Test Data Preparation Report (from data_prep.py)</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {"".join(html_parts)}
        <script>
            var coll = document.getElementsByClassName("collapsible");
            var i;
            for (i = 0; i < coll.length; i++) {{
                coll[i].addEventListener("click", function() {{
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.maxHeight){{ content.style.maxHeight = null; }} else {{ content.style.maxHeight = content.scrollHeight + "px"; }}
                }});
            }}
        </script>
    </body>
    </html>
    """
    with open(test_report_filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Test report from data_prep.py saved to: {test_report_filename}")