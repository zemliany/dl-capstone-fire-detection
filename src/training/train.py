#!/usr/bin/env python3

import tensorflow as tf
import yaml
import os

from src.data.manage import init_dataset, prepare_dataset_and_generators
from src.models.mobilenet_v2 import build_mobilenet_model, unfreeze_and_compile
from src.utils.report_generator import generate_full_report

def train_model(config_path='config/training_cfg.yaml', use_gpu=True):
    """
    Execute process of train model based on config

    :param config_path: config 
    :param use_gpu: Флаг для визначення, чи використовувати GPU.
    """

    if use_gpu:
        print("Init GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} PHY GPU, {len(logical_gpus)} LOG GPU")
            except RuntimeError as e:
                print(f"Error of init GPU: {e}. Switch to CPU.")
                use_gpu = False
        else:
            print("GPU not found. Switching to CPU.")
            use_gpu = False
    else:
        print("Learning will be executed based on CPU")
        tf.config.set_visible_devices([], 'GPU') 


    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Параметри з конфігурації
    dataset_name = config['dataset']['name']
    dataset_base_dir = config['dataset']['ds_local_path']
    img_height = config['model']['img_height']
    img_width = config['model']['img_width']
    batch_size = config['training']['batch_size']
    initial_epochs = config['training']['initial_epochs']
    fine_tune_epochs = config['training']['fine_tune_epochs']
    initial_learning_rate = config['training']['initial_learning_rate']
    fine_tune_learning_rate = config['training']['fine_tune_learning_rate']
    model_save_path = config['paths']['model_save_path']
    report_output_dir = 'reports'
    test_split_size = config['data_split']['test_size']
    validation_split_size = config['data_split']['validation_size']
    unfreeze_layers = config['training']['unfreeze_layers_from_end']

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(report_output_dir, exist_ok=True) # Створюємо папку reports

    print("--- DataSet Init ---")
    init_dataset(dataset_name, destination_dir=dataset_base_dir)
    print("--- DataSet Init completed ---")

    print("\n--- Start of data preparation and generator generation ---")
    train_generator, validation_generator, test_generator, dataset_report_html_parts = prepare_dataset_and_generators(
        dataset_base_dir=dataset_base_dir,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        test_size=test_split_size,
        validation_size=validation_split_size
    )
    print("--- Data and generator preparation completed ---")

    # dataset_summary_path = os.path.join(report_output_dir, 'dataset_summary.html')
    # with open(dataset_summary_path, 'w', encoding='utf-8') as f:
    #     f.write("".join(dataset_report_html_parts))
    # print(f"Dataset report saved to: {dataset_summary_path}")

    print("\n--- Build model ---")
    model = build_mobilenet_model(input_shape=(img_height, img_width, 3),
                                  num_classes=1,
                                  learning_rate=initial_learning_rate)
    model.summary()

    # Train model (feature extraction)
    print("\n--- Start of training (feature extraction) ---")
    callbacks_initial = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, 'best_model_initial.h5'),
                        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training']['reduce_lr_factor'],
                          patience=config['training']['reduce_lr_patience'], verbose=1, min_lr=config['training']['min_lr'])
    ]

    history_initial = model.fit(
        train_generator,
        epochs=initial_epochs,
        validation_data=validation_generator,
        callbacks=callbacks_initial
    )

    # Fine-tune
    print("\n--- Fine tune begin ---")
    model = unfreeze_and_compile(model, learning_rate=fine_tune_learning_rate, unfreeze_layers_from_end=unfreeze_layers)
    model.summary()

    callbacks_fine_tune = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, 'best_model_fine_tune.h5'),
                        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training']['reduce_lr_factor'],
                          patience=config['training']['reduce_lr_patience'], verbose=1, min_lr=config['training']['min_lr_fine_tune'])
    ]

    history_fine_tune = model.fit(
        train_generator,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=history_initial.epoch[-1] + 1,
        validation_data=validation_generator,
        callbacks=callbacks_fine_tune
    )
    print("--- Fine tune finished ---")

    print("\n--- Collecting data for the report and evaluating the model ---")
    
    test_generator.reset()

    predictions_proba = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred_proba = predictions_proba.flatten() 

    y_true = test_generator.classes

    y_pred_class = (y_pred_proba > 0.5).astype(int)

    # Loss and accuracy estimation (directly from the model)
    test_generator.reset()
    loss, accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Saving the final model in Keras format
    final_model_path = os.path.join(model_save_path, 'final_model.keras')
    model.save(final_model_path)
    print(f"Final model saved to in keras format: {final_model_path}")

    # REPORT
    print("\n--- Report Generation ---")

    # Unified learning history      
    full_history = {
        'loss': history_initial.history['loss'] + history_fine_tune.history['loss'],
        'accuracy': history_initial.history['accuracy'] + history_fine_tune.history['accuracy'],
        'val_loss': history_initial.history['val_loss'] + history_fine_tune.history['val_loss'],
        'val_accuracy': history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
    }
    

    full_report_filename = config['paths']['report_path'].split('/')[1]
    full_report_path = os.path.join(report_output_dir, full_report_filename)

    generate_full_report(
        report_filename=full_report_path,
        dataset_report_html_parts="".join(dataset_report_html_parts),
        config=config,
        history_data=full_history,
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        y_pred_class=y_pred_class,
        test_loss=loss,
        test_accuracy=accuracy
    )
    print(f"--- Report Generation Finished. Report available {full_report_path} ---")


# Optional: if you want to test this module independently
if __name__ == "__main__":
    print("Execute train.py for test...")
    train_model(use_gpu=False)