#!/usr/bin/env python3

import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import pickle
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import seaborn as sns

def plot_to_base64(plt_figure):
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt_figure.clear()
    plt.close(plt_figure)
    return image_base64

def generate_training_history_plots(history_data, initial_epochs=None):
    """
        Generates graphs of the training history (loss and accuracy) and returns them as base64.
        Takes a dictionary of history_data (the merged history from train.py).
        initial_epochs: Number of initial training epochs to mark the start of fine-tuning.
    """
    print("Generation of a learning history graph...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    epochs = range(len(history_data['accuracy']))
    
    # Plot Accuracy
    axes[0].plot(epochs, history_data['accuracy'], label='Training Accuracy')
    axes[0].plot(epochs, history_data['val_accuracy'], label='Validation Accuracy')
    
    if initial_epochs is not None and initial_epochs < len(epochs):
        axes[0].axvline(x=initial_epochs - 0.5, color='r', linestyle='--', label='Start of fine tuning')
    
    axes[0].set_title('Training: Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Precision')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(np.arange(0, len(epochs) + 1, max(1, len(epochs) // 10)))

    # Plot Loss
    axes[1].plot(epochs, history_data['loss'], label='Training Loss')
    axes[1].plot(epochs, history_data['val_loss'], label='Validation Loss')
    
    if initial_epochs is not None and initial_epochs < len(epochs):
        axes[1].axvline(x=initial_epochs - 0.5, color='r', linestyle='--', label='Start of fine tuning')
    
    axes[1].set_title('Learning: Losses')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(np.arange(0, len(epochs) + 1, max(1, len(epochs) // 10)))

    plt.suptitle("Model training history (Initial training + Fine tuning)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return plot_to_base64(fig)


def generate_roc_curve_plot(y_true, y_pred_proba):
    """
        ROC Curve => base64.
    """
    print("Perform ROC Curve...")
    if len(np.unique(y_true)) < 2:
        print("Warning: ROC Curve cannot be generated because y_true contains only one class np.unique < 2")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'ROC Curve cannot be generated\n(only one class in test data)', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('ROC Curve')
        return plot_to_base64(fig)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return plot_to_base64(fig)


def generate_confusion_matrix_plot(y_true, y_pred_class, class_names):
    """ 
        Confusion Matrix => base64
    """
    print("Generate Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred_class)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    return plot_to_base64(fig)


def generate_precision_recall_curve_plot(y_true, y_pred_proba):
    """ 
        Precision-recall => base64
    """
    print("Generate Precision-Recall Curve...")
    if len(np.unique(y_true)) < 2:
        print("Warn: Precision-Recall Curve cannot be generated because y_true contains only one class np.unique < 2")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Precision-Recall Curve cannot be generated', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Precision-Recall Curve')
        return plot_to_base64(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (AP = {ap:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    return plot_to_base64(fig)


def generate_full_report(report_filename: str, 
                         dataset_report_html_parts: str, 
                         config: dict, 
                         history_data: dict = None,
                         y_true: np.ndarray = None,
                         y_pred_proba: np.ndarray = None,
                         y_pred_class: np.ndarray = None,
                         test_loss: float = None,
                         test_accuracy: float = None):
    """
        Generate full HTML artifact
    """
    print(f"Start generating a full HTML report to a file: {report_filename}")

    report_output_dir = os.path.dirname(report_filename)
    if not report_output_dir: 
        report_output_dir = 'reports'

    if history_data is None:
        history_file_path = os.path.join(report_output_dir, 'training_history.pkl')
        if not os.path.exists(history_file_path):
            print(f"Error: Learning history file '{history_file_path}' not found.")
            return
        with open(history_file_path, 'rb') as f:
            history_data = pickle.load(f)

    if y_true is None or y_pred_proba is None or y_pred_class is None or test_loss is None or test_accuracy is None:
        eval_data_file_path = os.path.join(report_output_dir, 'evaluation_data.npz')
        if not os.path.exists(eval_data_file_path):
            print(f"Error: Assessment data file '{eval_data_file_path}' not found.")
            return
        eval_data = np.load(eval_data_file_path)
        y_true = eval_data['y_true']
        y_pred_proba = eval_data['y_pred_proba']
        y_pred_class = eval_data['y_pred_class']
        test_loss = eval_data['test_loss'].item()
        test_accuracy = eval_data['test_accuracy'].item()
        eval_data.close()

    initial_epochs = config.get('training', {}).get('initial_epochs', None)

    training_history_base64 = generate_training_history_plots(history_data, initial_epochs)
    roc_curve_base64 = generate_roc_curve_plot(y_true, y_pred_proba)
    
    class_names = ['Non-Fire', 'Fire']
    confusion_matrix_base64 = generate_confusion_matrix_plot(y_true, y_pred_class, class_names)
    precision_recall_base64 = generate_precision_recall_curve_plot(y_true, y_pred_proba)

    # HTML DATA
    report_body_content = ""

    # report_body_content += f"""
    # <button class="collapsible">Інформація про Датасет</button>
    # <div class="content">
    #     {dataset_report_html_parts}
    # </div>
    # """

    report_body_content += f"""  
        {dataset_report_html_parts}
    """

    training_results_html = f"""
    <button class="collapsible">Model Training Results</button>
    <div class="content">
        <h2>Metrics on the test data:</h2>
        <ul>
            <li><strong>Test Loss:</strong> {test_loss:.4f}</li>
            <li><strong>Test Accuracy:</strong> {test_accuracy:.4f}</li>
        </ul>

        <h2>Learning history:</h2>
        <div class="image-container">
            <img src='data:image/png;base64,{training_history_base64}' alt='Графіки історії навчання'>
        </div>

        <h2>ROC Curve:</h2>
        <div class="image-container">
            <img src='data:image/png;base64,{roc_curve_base64}' alt='ROC Curve'>
        </div>

        <h2>Confusion Matrix:</h2>
        <div class="image-container">
            <img src='data:image/png;base64,{confusion_matrix_base64}' alt='Confusion Matrix'>
        </div>

        <h2>Precision-Recall Curve:</h2>
        <div class="image-container">
            <img src='data:image/png;base64,{precision_recall_base64}' alt='Precision-Recall Curve'>
        </div>

        <h2>Параметри конфігурації:</h2>
        <pre>{yaml.dump(config, indent=2, sort_keys=False)}</pre>
    </div>
    """
    report_body_content += training_results_html


    final_html = f"""
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Full report on the fire detection project</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #e9f0f6; color: #333; line-height: 1.6; }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 25px; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            p {{ text-align: center; color: #555; font-size: 0.9em; }}
            h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}

            .collapsible {{
                background-color: #3498db;
                color: white;
                cursor: pointer;
                padding: 15px 20px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 1.1em;
                transition: 0.4s;
                margin-top: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .active, .collapsible:hover {{
                background-color: #2980b9;
            }}
            .collapsible:after {{
                content: '\\02795';
                font-size: 0.8em;
                color: white;
                float: right;
                margin-left: 5px;
            }}
            .active:after {{
                content: '\\2796';
            }}
            .content {{
                padding: 0 18px;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
                background-color: #ffffff;
                border-left: 1px solid #ddd;
                border-right: 1px solid #ddd;
                border-bottom: 1px solid #ddd;
                border-radius: 0 0 5px 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .content pre {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-size: 0.9em;
                border: 1px solid #eee;
            }}
            .image-container {{
                text-align: center;
                margin-bottom: 20px;
                background-color: #fdfdfd;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #eee;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 15px auto;
                border: 1px solid #ccc;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Full report on the fire detection project</h1>
        <p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        {report_body_content}

        <script>
            var coll = document.getElementsByClassName("collapsible");
            var i;

            for (i = 0; i < coll.length; i++) {{
                coll[i].addEventListener("click", function() {{
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.maxHeight){{
                        content.style.maxHeight = null;
                    }} else {{
                        content.style.maxHeight = content.scrollHeight + "px";
                    }}
                }});
            }}
        </script>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"\nFull : {report_filename}")


if __name__ == "__main__":
    print("Running the report_generator.py module for testing (generating mock data in 'reports/')...")

    test_report_dir = 'reports' 
    os.makedirs(test_report_dir, exist_ok=True)

    # MOCK DATA for standalone script test without training model
    epochs_initial = 5
    epochs_fine_tune = 10
    total_epochs = epochs_initial + epochs_fine_tune
    
    mock_full_history_data = {
        'accuracy': np.linspace(0.5, 0.75, total_epochs).tolist(),
        'val_accuracy': np.linspace(0.65, 0.92, total_epochs).tolist(),
        'loss': np.linspace(0.5, 0.1, total_epochs).tolist(),
        'val_loss': np.linspace(0.6, 0.15, total_epochs).tolist(),
    }

    # Save mock history to file for testing purposes if generate_full_report loads from file
    with open(os.path.join(test_report_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(mock_full_history_data, f)
        
    # Mock evaluation data (y_true, y_pred_proba, y_pred_class)
    num_samples = 200 
    mock_y_true = np.random.randint(0, 2, num_samples)
    mock_y_pred_proba = np.clip(mock_y_true + np.random.normal(0, 0.3, num_samples), 0, 1)
    mock_y_pred_class = (mock_y_pred_proba > 0.5).astype(int)

    mock_test_loss = 0.0384
    mock_test_accuracy = 0.9800

    np.savez(os.path.join(test_report_dir, 'evaluation_data.npz'),
             y_true=mock_y_true,
             y_pred_proba=mock_y_pred_proba,
             y_pred_class=mock_y_pred_class,
             test_loss=mock_test_loss,
             test_accuracy=mock_test_accuracy)
             
    # Mock dataset summary HTML
    mock_dataset_summary_html = """
    <h3>Тестова інформація про датасет</h3>
    <p>Це демонстраційна інформація, згенерована для тестування.</p>
    <ul>
        <li>Загальна кількість зображень: 1000</li>
        <li>Кількість зображень для навчання: 700</li>
        <li>Кількість зображень для валідації: 150</li>
        <li>Кількість зображень для тестування: 150</li>
    </ul>
    """
    with open(os.path.join(test_report_dir, 'dataset_summary.html'), 'w', encoding='utf-8') as f:
        f.write(mock_dataset_summary_html)

    config_dir_for_test = 'config'
    os.makedirs(config_dir_for_test, exist_ok=True)
    mock_config_path = os.path.join(config_dir_for_test, 'training_config.yaml')
    
    mock_config = {
        'dataset': {'name': 'test_dataset', 'ds_local_path': 'test/path'},
        'model': {'img_height': 224, 'img_width': 224, 'num_classes': 1},
        'data_split': {'test_size': 0.15, 'validation_size': 0.15},
        'training': {
            'batch_size': 32, 'initial_epochs': epochs_initial, 'fine_tune_epochs': epochs_fine_tune,
            'initial_learning_rate': 0.001, 'fine_tune_learning_rate': 0.0001,
            'early_stopping_patience': 10, 'reduce_lr_factor': 0.2,
            'reduce_lr_patience': 5, 'min_lr': 1e-06, 'min_lr_fine_tune': 1e-07,
            'unfreeze_layers_from_end': 50
        },
        'paths': {'model_save_path': 'test_models', 'report_file_name': 'full_training_report_test_direct.html'}
    }
    with open(mock_config_path, 'w') as f:
        yaml.dump(mock_config, f, indent=2, sort_keys=False)

    generate_full_report(
        report_filename=os.path.join(test_report_dir, 'full_test_report_direct.html'),
        dataset_report_html_parts=mock_dataset_summary_html,
        config=mock_config,
        history_data=mock_full_history_data,
        y_true=mock_y_true,
        y_pred_proba=mock_y_pred_proba,
        y_pred_class=mock_y_pred_class,
        test_loss=mock_test_loss,
        test_accuracy=mock_test_accuracy
    )

    print(f"\nTest report and mock graphs generated in folder'{test_report_dir}/'.")