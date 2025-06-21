# Speech Emotion Recognition

A deep learning project for recognizing emotions from speech audio using the Wav2Vec2 transformer model. This system can classify audio into seven different emotional categories: fear, anger, disgust, neutral, sadness, pleasant surprise, and happiness.

## Features

- **Multi-emotion Classification**: Recognizes 7 different emotions from speech
- **Real-time Audio Processing**: Record and analyze live audio with keyboard controls
- **Pre-trained Model Support**: Uses Facebook's Wav2Vec2 as the base model
- **Batch Processing**: Process multiple audio files or chunks simultaneously
- **Audio Visualization**: Generate waveforms and spectrograms for analysis
- **Flexible Input**: Supports both file-based and live audio input

## Supported Emotions

1. **Fear** - Anxious or frightened speech patterns
2. **Anger** - Aggressive or hostile vocal expressions
3. **Disgust** - Expressions of revulsion or distaste
4. **Neutral** - Calm, emotionally neutral speech
5. **Sadness** - Melancholic or sorrowful vocal patterns
6. **Pleasant Surprise** - Positive, surprised expressions
7. **Happiness** - Joyful, cheerful speech patterns

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

**Download the dataset & model here:**  
- [Dataset Link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)  
- [Trained Model Link](https://drive.google.com/drive/folders/1HwVmJ7UBVWYBLBnt45iAXaciN9xIl-Zq)

### Dependencies

Install the required packages using pip:

```bash
pip install torch torchaudio transformers
pip install librosa soundfile sounddevice
pip install pandas numpy matplotlib seaborn
pip install scikit-learn pydub pynput
pip install jupyter ipython
```

### Clone the Repository

```bash
git clone <repository-url>
cd speech-emotion-recognition
```

## Dataset

This project uses the **TESS (Toronto Emotional Speech Set)** dataset, which contains:
- 2,800 audio files
- 7 emotional categories
- Multiple speakers
- Consistent audio quality and format

### Dataset Structure
```
dataset/
├── TESS Toronto emotional speech set data/
│   ├── YAF_angry/
│   ├── YAF_disgust/
│   ├── YAF_fear/
│   ├── YAF_happy/
│   ├── YAF_neutral/
│   ├── YAF_pleasant_surprised/
│   └── YAF_sad/
```

## Usage

### 1. Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook "Speech Emotion Recognition - Sound Classification/speech_emotion_recognition.ipynb"
```

The notebook includes:
- Data loading and preprocessing
- Exploratory data analysis with visualizations
- Model training with Wav2Vec2
- Evaluation metrics and results

### 2. Real-time Emotion Recognition

Run the real-time prediction script:

```bash
python final_test.py
```

**Controls:**
- Press **'p'** to pause recording
- Press **'r'** to resume recording  
- Press **'e'** to end recording and get results

### 3. Single File Prediction

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa

# Load the trained model
processor = Wav2Vec2Processor.from_pretrained("path/to/trained_model")
model = Wav2Vec2ForSequenceClassification.from_pretrained("path/to/trained_model")

def predict_emotion(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label

# Example usage
emotion_id = predict_emotion("path/to/audio/file.wav")
print(f"Predicted emotion: {emotion_mapping[emotion_id]}")
```

## Model Architecture

- **Base Model**: Facebook's Wav2Vec2-base
- **Classification Head**: Custom sequence classification layer
- **Input**: 16kHz audio waveforms
- **Output**: 7-class emotion probabilities
- **Training**: Fine-tuned on TESS dataset

### Model Performance

The model achieves competitive performance on emotion recognition tasks with metrics including:
- Accuracy
- Precision
- Recall  
- F1-score

## File Structure

```
├── Speech Emotion Recognition - Sound Classification/
│   └── speech_emotion_recognition.ipynb    # Main training notebook
├── final_test.py                          # Real-time prediction script
├── dataset/                               # Audio dataset directory
├── results/                               # Training outputs and checkpoints
├── trained_model/                         # Saved model files
└── README.md                             # This file
```

## Technical Details

### Audio Processing
- **Sample Rate**: 16kHz (required by Wav2Vec2)
- **Max Length**: 32,000 samples (~2 seconds)
- **Preprocessing**: Padding/truncation, normalization
- **Feature Extraction**: Wav2Vec2 transformer embeddings

### Training Configuration
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay
- **Evaluation**: Per-epoch validation

## Real-time Processing Features

- **Chunk-based Analysis**: Processes audio in 3-second segments
- **Keyboard Controls**: Interactive recording control
- **Percentage Output**: Shows emotion distribution across chunks
- **Memory Efficient**: Uses in-memory buffers for processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- Librosa 0.8+
- NumPy, Pandas, Matplotlib
- Jupyter Notebook (for training)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face Transformers** for the Wav2Vec2 implementation
- **TESS Dataset** creators for the emotional speech data
- **Facebook AI Research** for the original Wav2Vec2 model
- **Librosa** team for audio processing utilities

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training arguments
2. **Audio Loading Errors**: Ensure audio files are in supported formats (WAV, MP3)
3. **Model Loading Issues**: Check that the trained model path is correct
4. **Real-time Recording Problems**: Verify microphone permissions and audio device settings

### Performance Tips

- Use GPU acceleration for faster training
- Preprocess audio files to consistent format and sample rate
- Consider data augmentation for better generalization
- Monitor training metrics to prevent overfitting

## Future Enhancements

- [ ] Support for additional emotion categories
- [ ] Multi-language emotion recognition
- [ ] Web interface for easy interaction
- [ ] Mobile app integration
- [ ] Continuous learning capabilities
- [ ] Speaker-independent improvements
