#!/usr/bin/env python3
"""
Test script to compare model predictions between command line and API methods
"""
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os

# Test file
test_file = "audio/aka2.flac"
model_path = "model/model-1.h5"

print("=" * 60)
print("Testing Model Predictions Comparison")
print("=" * 60)

# Load model
print("\nLoading model...")
model = tf.keras.models.load_model(model_path)

# Method 1: Command line method
print("\n1. Command Line Method:")
audio1, sr1 = librosa.load(test_file, sr=None)
print(f"   Audio: length={len(audio1)}, sr={sr1}")

mel1 = librosa.feature.melspectrogram(y=audio1, n_mels=91)
mel1 = librosa.power_to_db(mel1, ref=np.max)
print(f"   Mel shape before: {mel1.shape}")

if mel1.shape[1] < 150:
    mel1 = np.pad(mel1, ((0, 0), (0, 150 - mel1.shape[1])), mode='constant')
else:
    mel1 = mel1[:, :150]
print(f"   Mel shape after: {mel1.shape}")

# Command line way: array of arrays, then add channel
x1 = np.array([mel1])  # (1, 91, 150)
print(f"   Array shape before channel: {x1.shape}")
x1 = x1[..., np.newaxis]  # (1, 91, 150, 1)
print(f"   Final input shape: {x1.shape}")

pred1 = model.predict(x1, verbose=0)
print(f"   Predictions: {pred1}")
print(f"   AI: {pred1[0][0]:.6f} ({pred1[0][0]*100:.1f}%), Human: {pred1[0][1]:.6f} ({pred1[0][1]*100:.1f}%)")

# Method 2: API method
print("\n2. API Method:")
with open(test_file, 'rb') as f:
    file_bytes = f.read()

with tempfile.NamedTemporaryFile(mode='wb', suffix='.flac', delete=False) as temp_file:
    temp_file.write(file_bytes)
    temp_file_path = temp_file.name

try:
    audio2, sr2 = librosa.load(temp_file_path, sr=None)
    print(f"   Audio: length={len(audio2)}, sr={sr2}")
    
    mel2 = librosa.feature.melspectrogram(y=audio2, n_mels=91)
    mel2 = librosa.power_to_db(mel2, ref=np.max)
    print(f"   Mel shape before: {mel2.shape}")
    
    if mel2.shape[1] < 150:
        mel2 = np.pad(mel2, ((0, 0), (0, 150 - mel2.shape[1])), mode='constant')
    else:
        mel2 = mel2[:, :150]
    print(f"   Mel shape after: {mel2.shape}")
    
    # API way: direct reshape
    x2 = mel2[np.newaxis, ..., np.newaxis]  # (1, 91, 150, 1)
    print(f"   Final input shape: {x2.shape}")
    
    pred2 = model.predict(x2, verbose=0)
    print(f"   Predictions: {pred2}")
    print(f"   AI: {pred2[0][0]:.6f} ({pred2[0][0]*100:.1f}%), Human: {pred2[0][1]:.6f} ({pred2[0][1]*100:.1f}%)")
finally:
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

# Compare
print("\n3. Comparison:")
print(f"   Input shapes match: {x1.shape == x2.shape}")
print(f"   Input arrays equal: {np.allclose(x1, x2)}")
if x1.shape == x2.shape:
    print(f"   Max input difference: {np.max(np.abs(x1 - x2))}")
print(f"   Predictions match: {np.allclose(pred1, pred2)}")
if pred1.shape == pred2.shape:
    print(f"   Max prediction difference: {np.max(np.abs(pred1 - pred2))}")
