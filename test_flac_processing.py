#!/usr/bin/env python3
"""
Test script to compare FLAC processing between command line and API methods
"""
import librosa
import numpy as np
import tempfile
import os

# Test file
test_file = "audio/aka2.flac"

print("=" * 60)
print("Testing FLAC Processing Comparison")
print("=" * 60)

# Method 1: Direct file load (command line method)
print("\n1. Command Line Method (direct file load):")
audio1, sr1 = librosa.load(test_file, sr=None)
print(f"   Audio length: {len(audio1)}, Sample rate: {sr1}")

mel1 = librosa.feature.melspectrogram(y=audio1, n_mels=91)
mel1 = librosa.power_to_db(mel1, ref=np.max)
print(f"   Mel shape before: {mel1.shape}")

if mel1.shape[1] < 150:
    mel1 = np.pad(mel1, ((0, 0), (0, 150 - mel1.shape[1])), mode='constant')
else:
    mel1 = mel1[:, :150]
print(f"   Mel shape after: {mel1.shape}")

# Method 2: Load from bytes via temp file (API method)
print("\n2. API Method (bytes -> temp file -> load):")
with open(test_file, 'rb') as f:
    file_bytes = f.read()
print(f"   File bytes size: {len(file_bytes)} bytes")

with tempfile.NamedTemporaryFile(mode='wb', suffix='.flac', delete=False) as temp_file:
    temp_file.write(file_bytes)
    temp_file_path = temp_file.name

try:
    audio2, sr2 = librosa.load(temp_file_path, sr=None)
    print(f"   Audio length: {len(audio2)}, Sample rate: {sr2}")
    
    mel2 = librosa.feature.melspectrogram(y=audio2, n_mels=91)
    mel2 = librosa.power_to_db(mel2, ref=np.max)
    print(f"   Mel shape before: {mel2.shape}")
    
    if mel2.shape[1] < 150:
        mel2 = np.pad(mel2, ((0, 0), (0, 150 - mel2.shape[1])), mode='constant')
    else:
        mel2 = mel2[:, :150]
    print(f"   Mel shape after: {mel2.shape}")
finally:
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

# Compare
print("\n3. Comparison:")
print(f"   Audio lengths match: {len(audio1) == len(audio2)}")
print(f"   Sample rates match: {sr1 == sr2}")
print(f"   Mel shapes match: {mel1.shape == mel2.shape}")
print(f"   Mel arrays equal: {np.allclose(mel1, mel2)}")
print(f"   Max difference: {np.max(np.abs(mel1 - mel2)) if mel1.shape == mel2.shape else 'N/A'}")

if not np.allclose(mel1, mel2):
    print("\n   WARNING: Mel-spectrograms differ!")
    print(f"   First 5x5 of mel1:\n{mel1[:5, :5]}")
    print(f"   First 5x5 of mel2:\n{mel2[:5, :5]}")
