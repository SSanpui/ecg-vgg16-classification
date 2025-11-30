"""
ECG Arrhythmia Classification - NO DATA LEAKAGE
Split FIRST, then augment ONLY training data

This script implements the RIGOROUS protocol:
1. Split original dataset into train/val/test (80/10/10)
2. Apply augmentation ONLY to training set
3. Evaluate on unmodified validation and test sets

Author: Based on SSanpui/ecg-vgg16-classification
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for ECG classification with NO data leakage"""
    
    # Data paths - MODIFY THESE
    DATASET_DIR = '/path/to/your/kaggle/ECG/dataset'  # Your Kaggle ECG dataset
    OUTPUT_DIR = './outputs_no_leakage'
    MODEL_DIR = './models_no_leakage'
    
    # Image specifications
    IMAGE_SIZE = (128, 128)
    CHANNELS = 3
    INPUT_SHAPE = (*IMAGE_SIZE, CHANNELS)
    
    # Classes
    NUM_CLASSES = 6
    CLASS_NAMES = ['Normal', 'LBBB', 'RBBB', 'PAC', 'PVC', 'VF']
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Data split (BEFORE augmentation)
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Augmentation parameters (ONLY for training set)
    AUGMENTATION_PARAMS = {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

# Set random seeds
np.random.seed(Config.RANDOM_SEED)
tf.random.set_seed(Config.RANDOM_SEED)

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING AND SPLITTING (NO LEAKAGE)
# ============================================================================

def load_original_dataset(dataset_dir):
    """
    Load original UNAUGMENTED dataset with labels
    
    Expected structure:
    dataset_dir/
        Normal/
            image1.png
            image2.png
        LBBB/
            image1.png
        ...
    """
    print("\n" + "="*80)
    print("LOADING ORIGINAL DATASET (NO AUGMENTATION YET)")
    print("="*80)
    
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(Config.CLASS_NAMES):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"WARNING: Directory {class_dir} not found!")
            continue
        
        class_images = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in class_images:
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_idx)
        
        print(f"{class_name}: {len(class_images)} images")
    
    print(f"\nTotal images: {len(image_paths)}")
    
    return np.array(image_paths), np.array(labels)

def split_dataset_stratified(image_paths, labels, config):
    """
    Split dataset into train/val/test BEFORE any augmentation
    This ensures NO DATA LEAKAGE
    
    Returns: train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    """
    print("\n" + "="*80)
    print("SPLITTING DATASET (BEFORE AUGMENTATION)")
    print("="*80)
    
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_SEED
    )
    
    # Second split: separate validation from training
    val_size_adjusted = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=config.RANDOM_SEED
    )
    
    print(f"\nDataset split (ORIGINAL, unaugmented):")
    print(f"Training set: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"Validation set: {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"Test set: {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    # Print class distribution
    print("\nClass distribution in each set:")
    for split_name, split_labels in [('Train', train_labels), 
                                      ('Val', val_labels), 
                                      ('Test', test_labels)]:
        print(f"\n{split_name}:")
        for class_idx, class_name in enumerate(Config.CLASS_NAMES):
            count = np.sum(split_labels == class_idx)
            print(f"  {class_name}: {count} images")
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

# ============================================================================
# DATA GENERATORS (Augmentation ONLY on Training)
# ============================================================================

def create_data_generators(train_paths, val_paths, test_paths, 
                          train_labels, val_labels, test_labels, config):
    """
    Create data generators with augmentation ONLY for training set
    Validation and test sets use ORIGINAL images without augmentation
    """
    print("\n" + "="*80)
    print("CREATING DATA GENERATORS")
    print("="*80)
    
    # TRAINING: With augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        **config.AUGMENTATION_PARAMS
    )
    
    # VALIDATION & TEST: NO augmentation (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create DataFrame for flow_from_dataframe
    train_df = pd.DataFrame({
        'filename': train_paths,
        'class': [config.CLASS_NAMES[label] for label in train_labels]
    })
    
    val_df = pd.DataFrame({
        'filename': val_paths,
        'class': [config.CLASS_NAMES[label] for label in val_labels]
    })
    
    test_df = pd.DataFrame({
        'filename': test_paths,
        'class': [config.CLASS_NAMES[label] for label in test_labels]
    })
    
    # Training generator (with augmentation)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # Validation generator (NO augmentation)
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Test generator (NO augmentation)
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='class',
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✓ Training generator: {len(train_generator)} batches (WITH augmentation)")
    print(f"✓ Validation generator: {len(val_generator)} batches (NO augmentation)")
    print(f"✓ Test generator: {len(test_generator)} batches (NO augmentation)")
    
    return train_generator, val_generator, test_generator

# ============================================================================
# MODEL ARCHITECTURE (Exact match to your paper)
# ============================================================================

def build_vgg16_model(config):
    """
    Build VGG16 transfer learning model
    Architecture: VGG16 → GlobalAveragePooling2D → Dropout(0.5) → Dense(6, softmax)
    
    This matches your paper's architecture exactly
    """
    print("\n" + "="*80)
    print("BUILDING VGG16 MODEL")
    print("="*80)
    
    # Load VGG16 with ImageNet weights, without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model with YOUR exact architecture
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Replace FC layers
        layers.Dropout(0.5),               # Regularization
        layers.Dense(config.NUM_CLASSES, activation='softmax')  # Output layer
    ], name='VGG16_Transfer_Learning')
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Parameters reduced from 138M to {total_params/1e6:.1f}M")
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_gen, val_gen, config):
    """Train model with early stopping and learning rate reduction"""
    
    print("\n" + "="*80)
    print("TRAINING MODEL (NO DATA LEAKAGE)")
    print("="*80)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, 'best_model_no_leakage.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    
    return history, training_time

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_gen, config):
    """Comprehensive evaluation on test set"""
    
    print("\n" + "="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    # Evaluate
    test_results = model.evaluate(test_gen, verbose=1)
    
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_precision = test_results[2]
    test_recall = test_results[3]
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    # Detailed predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix (No Data Leakage)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix_no_leakage.png'), dpi=300)
    print(f"\n✓ Confusion matrix saved")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }

# ============================================================================
# CROSS-VALIDATION (No Data Leakage)
# ============================================================================

def stratified_k_fold_cv(image_paths, labels, config, k=5):
    """
    Perform stratified k-fold cross-validation
    CRITICAL: Each fold splits BEFORE augmentation
    """
    print("\n" + "="*80)
    print(f"STRATIFIED {k}-FOLD CROSS-VALIDATION (NO DATA LEAKAGE)")
    print("="*80)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{k}")
        print(f"{'='*80}")
        
        # Split data for this fold
        fold_train_paths = image_paths[train_idx]
        fold_val_paths = image_paths[val_idx]
        fold_train_labels = labels[train_idx]
        fold_val_labels = labels[val_idx]
        
        print(f"Training: {len(fold_train_paths)} images")
        print(f"Validation: {len(fold_val_paths)} images")
        
        # Create generators (augmentation ONLY on training)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **config.AUGMENTATION_PARAMS
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_df = pd.DataFrame({
            'filename': fold_train_paths,
            'class': [config.CLASS_NAMES[label] for label in fold_train_labels]
        })
        
        val_df = pd.DataFrame({
            'filename': fold_val_paths,
            'class': [config.CLASS_NAMES[label] for label in fold_val_labels]
        })
        
        fold_train_gen = train_datagen.flow_from_dataframe(
            train_df, x_col='filename', y_col='class',
            target_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE,
            class_mode='categorical', shuffle=True, seed=config.RANDOM_SEED
        )
        
        fold_val_gen = val_datagen.flow_from_dataframe(
            val_df, x_col='filename', y_col='class',
            target_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE,
            class_mode='categorical', shuffle=False
        )
        
        # Build and train model
        fold_model = build_vgg16_model(config)
        
        fold_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        fold_history = fold_model.fit(
            fold_train_gen,
            validation_data=fold_val_gen,
            epochs=config.EPOCHS,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, 
                            restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                patience=5, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )
        
        # Evaluate
        fold_results = fold_model.evaluate(fold_val_gen, verbose=0)
        
        accuracy = fold_results[1]
        precision = fold_results[2]
        recall = fold_results[3]
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        cv_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
    
    # Summary statistics
    cv_df = pd.DataFrame(cv_results)
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nMean ± Std:")
    print(f"  Accuracy: {cv_df['accuracy'].mean()*100:.1f}% ± {cv_df['accuracy'].std()*100:.1f}%")
    print(f"  Precision: {cv_df['precision'].mean():.3f} ± {cv_df['precision'].std():.3f}")
    print(f"  Recall: {cv_df['recall'].mean():.3f} ± {cv_df['recall'].std():.3f}")
    print(f"  F1-Score: {cv_df['f1_score'].mean():.3f} ± {cv_df['f1_score'].std():.3f}")
    
    # Save results
    cv_df.to_csv(os.path.join(config.OUTPUT_DIR, 'cv_results_no_leakage.csv'), index=False)
    
    return cv_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    config = Config()
    
    print("\n" + "="*80)
    print("ECG ARRHYTHMIA CLASSIFICATION - NO DATA LEAKAGE")
    print("Split FIRST, Augment ONLY Training Set")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.DATASET_DIR}")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Split: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}")
    
    # Load original dataset
    image_paths, labels = load_original_dataset(config.DATASET_DIR)
    
    # Split dataset (BEFORE augmentation)
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        split_dataset_stratified(image_paths, labels, config)
    
    # Create generators (augmentation ONLY on training)
    train_gen, val_gen, test_gen = create_data_generators(
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels,
        config
    )
    
    # Build model
    model = build_vgg16_model(config)
    
    # Train model
    history, training_time = train_model(model, train_gen, val_gen, config)
    
    # Evaluate model
    test_results = evaluate_model(model, test_gen, config)
    
    # Save model
    model.save(os.path.join(config.MODEL_DIR, 'final_model_no_leakage.h5'))
    print(f"\n✓ Model saved to {config.MODEL_DIR}")
    
    # Save results summary
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'training_time_hours': training_time / 3600,
        'test_accuracy': float(test_results['test_accuracy']),
        'test_precision': float(test_results['test_precision']),
        'test_recall': float(test_results['test_recall']),
        'test_f1': float(test_results['test_f1']),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'augmentation': 'ONLY on training set (NO DATA LEAKAGE)'
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {config.OUTPUT_DIR}")
    
    # Optional: Run cross-validation
    print("\n" + "="*80)
    print("Do you want to run 5-fold cross-validation? (y/n)")
    print("(This will take additional time)")
    print("="*80)
    
    # For automated run, uncomment:
    # cv_results = stratified_k_fold_cv(image_paths, labels, config, k=5)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Accuracy (NO DATA LEAKAGE): {test_results['test_accuracy']*100:.2f}%")
    print(f"Compare this with your original 99.0% to quantify data leakage impact")

if __name__ == "__main__":
    main()
