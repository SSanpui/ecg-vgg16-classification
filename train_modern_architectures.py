"""
ECG Arrhythmia Classification - Modern Architecture Comparison
Compare VGG16 with ResNet50, EfficientNet, DenseNet, MobileNet, ViT

All models trained on IDENTICAL data splits with NO data leakage
Author: Based on SSanpui/ecg-vgg16-classification
"""

import os
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    VGG16, ResNet50, ResNet18, EfficientNetB0, EfficientNetB1,
    DenseNet121, MobileNetV2, InceptionV3
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# CONFIGURATION (Same as your VGG16 training)
# ============================================================================

class Config:
    # Data paths - MODIFY THESE
    TRAIN_DIR = './split_data/train'  # Created by train_no_leakage.py
    VAL_DIR = './split_data/val'
    TEST_DIR = './split_data/test'
    
    OUTPUT_DIR = './outputs_comparison'
    MODEL_DIR = './models_comparison'
    
    # Image specs
    IMAGE_SIZE = (128, 128)
    INPUT_SHAPE = (*IMAGE_SIZE, 3)
    
    # Classes
    NUM_CLASSES = 6
    CLASS_NAMES = ['Normal', 'LBBB', 'RBBB', 'PAC', 'PVC', 'VF']
    
    # Training (IDENTICAL to VGG16)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Augmentation (ONLY on training)
    AUGMENTATION_PARAMS = {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True
    }
    
    RANDOM_SEED = 42

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)

tf.random.set_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ============================================================================
# DATA GENERATORS (IDENTICAL for all models)
# ============================================================================

def create_generators(config):
    """Create identical data generators for fair comparison"""
    
    # Training: With augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        **config.AUGMENTATION_PARAMS
    )
    
    # Val/Test: NO augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

# ============================================================================
# MODEL BUILDING FUNCTIONS
# ============================================================================

def build_vgg16(input_shape, num_classes):
    """Your original VGG16 architecture"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='VGG16')
    
    return model

def build_resnet50(input_shape, num_classes):
    """ResNet50 with same top structure"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='ResNet50')
    
    return model

def build_resnet18(input_shape, num_classes):
    """ResNet18 - lighter version"""
    # Note: ResNet18 not in keras.applications, using ResNet50V2 as proxy
    from tensorflow.keras.applications import ResNet50V2
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='ResNet18')
    
    return model

def build_efficientnetb0(input_shape, num_classes):
    """EfficientNetB0 - parameter efficient"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='EfficientNetB0')
    
    return model

def build_efficientnetb1(input_shape, num_classes):
    """EfficientNetB1 - slightly larger"""
    base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='EfficientNetB1')
    
    return model

def build_densenet121(input_shape, num_classes):
    """DenseNet121"""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='DenseNet121')
    
    return model

def build_mobilenetv2(input_shape, num_classes):
    """MobileNetV2 - mobile/edge deployment"""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='MobileNetV2')
    
    return model

def build_inceptionv3(input_shape, num_classes):
    """InceptionV3"""
    # InceptionV3 requires minimum 75x75 input
    if input_shape[0] < 75:
        print(f"WARNING: InceptionV3 requires min 75x75, got {input_shape[0]}x{input_shape[1]}")
        print("Resizing to 75x75 for InceptionV3")
        input_shape = (75, 75, 3)
    
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='InceptionV3')
    
    return model

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(model_builder, model_name, train_gen, val_gen, test_gen, config):
    """
    Train and evaluate a single model
    Returns dict with all metrics for Table 6
    """
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name}")
    print(f"{'='*80}\n")
    
    # Build model
    model = model_builder(config.INPUT_SHAPE, config.NUM_CLASSES)
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"Parameters:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    # Train
    print("\nTraining...")
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    training_time_hrs = training_time / 3600
    
    print(f"\nTraining completed in {training_time_hrs:.1f} hours")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_gen.reset()
    test_results = model.evaluate(test_gen, verbose=1)
    
    test_loss = test_results[0]
    test_acc = test_results[1]
    test_prec = test_results[2]
    test_rec = test_results[3]
    test_f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-7)
    
    # Measure inference time
    print("\nMeasuring inference time...")
    sample_batch = next(iter(test_gen))[0]
    
    # Warm-up
    _ = model.predict(sample_batch[:1], verbose=0)
    
    # Time 100 single-image inferences
    inference_times = []
    for i in range(min(100, len(sample_batch))):
        start = time.time()
        _ = model.predict(sample_batch[i:i+1], verbose=0)
        inference_times.append(time.time() - start)
    
    avg_inference_ms = np.mean(inference_times) * 1000
    
    # Compile results
    results = {
        'Architecture': model_name,
        'Parameters_M': round(total_params / 1e6, 1),
        'Accuracy': f"{test_acc*100:.1f}%",
        'Precision': f"{test_prec*100:.1f}%",
        'Recall': f"{test_rec*100:.1f}%",
        'F1_Score': f"{test_f1*100:.1f}%",
        'Training_Time_hrs': round(training_time_hrs, 1),
        'Inference_Time_ms': round(avg_inference_ms, 0),
        # Raw values for comparison
        'accuracy_raw': float(test_acc),
        'f1_raw': float(test_f1)
    }
    
    # Save model
    model_path = os.path.join(config.MODEL_DIR, f'{model_name}_best.h5')
    model.save(model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    return results

# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    """Run comparison of all architectures"""
    
    config = Config()
    
    print("\n" + "="*80)
    print("MODERN ARCHITECTURE COMPARISON")
    print("All models trained on IDENTICAL data (NO data leakage)")
    print("="*80)
    
    # Load data generators (SAME for all models)
    print("\nCreating data generators...")
    train_gen, val_gen, test_gen = create_generators(config)
    
    # List of architectures to compare
    architectures = [
        (build_vgg16, 'VGG16'),
        (build_resnet50, 'ResNet50'),
        (build_efficientnetb0, 'EfficientNetB0'),
        (build_efficientnetb1, 'EfficientNetB1'),
        (build_densenet121, 'DenseNet121'),
        (build_mobilenetv2, 'MobileNetV2'),
        (build_inceptionv3, 'InceptionV3'),
    ]
    
    all_results = []
    
    for model_builder, model_name in architectures:
        try:
            results = train_and_evaluate(
                model_builder, model_name,
                train_gen, val_gen, test_gen,
                config
            )
            all_results.append(results)
            
            # Save intermediate results
            df_temp = pd.DataFrame(all_results)
            df_temp.to_csv(
                os.path.join(config.OUTPUT_DIR, 'comparison_intermediate.csv'),
                index=False
            )
            
        except Exception as e:
            print(f"\n❌ ERROR training {model_name}: {e}")
            print("Continuing with next architecture...")
            continue
    
    # Create final results table
    df_final = pd.DataFrame(all_results)
    
    # Sort by accuracy (descending)
    df_final = df_final.sort_values('accuracy_raw', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS - TABLE 6")
    print("="*80 + "\n")
    
    display_df = df_final[['Architecture', 'Parameters_M', 'Accuracy', 
                           'Precision', 'Recall', 'F1_Score',
                           'Training_Time_hrs', 'Inference_Time_ms']]
    
    print(display_df.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(config.OUTPUT_DIR, 'table6_comparison_results.csv')
    display_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")
    
    # Save to LaTeX
    latex_table = display_df.to_latex(index=False, escape=False)
    latex_path = os.path.join(config.OUTPUT_DIR, 'table6_latex.txt')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved: {latex_path}")
    
    # Save full results with metadata
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'image_size': config.IMAGE_SIZE
        },
        'results': all_results
    }
    
    json_path = os.path.join(config.OUTPUT_DIR, 'full_comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Full results saved: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_acc = df_final.iloc[0]
    print(f"\nBest Accuracy: {best_acc['Architecture']} - {best_acc['Accuracy']}")
    print(f"Your VGG16: {df_final[df_final['Architecture']=='VGG16']['Accuracy'].values[0]}")
    
    print(f"\nMost Parameter Efficient:")
    most_efficient = df_final.loc[df_final['Parameters_M'].idxmin()]
    print(f"  {most_efficient['Architecture']}: {most_efficient['Parameters_M']}M params, {most_efficient['Accuracy']} accuracy")
    
    print(f"\nFastest Inference:")
    fastest = df_final.loc[df_final['Inference_Time_ms'].idxmin()]
    print(f"  {fastest['Architecture']}: {fastest['Inference_Time_ms']}ms")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nUse these results for Table 6 in your revised paper")
    print(f"All models trained with NO data leakage for fair comparison")

if __name__ == "__main__":
    main()
