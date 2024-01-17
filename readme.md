# Gender Recognition Algorithm Based on Voice

## Algorithm Description

This algorithm for gender recognition from voice samples uses signal processing techniques based on the Harmonic Product Spectrum (HPS). It analyzes the frequency characteristics of male and female voices.

### Frequency Ranges:
- Male: 55-160 Hz
- Female: 170-275 Hz

### Harmonic Product Spectrum (HPS):
- Time Frame Length: 3 seconds
- Iteration Limit: 5
- Implementation Details:
  - Divide the signal into frames of 1-second length.
  - For each frame:
    - Apply Hamming window.
    - Perform FFT (Fast Fourier Transform).
    - Multiply FFT values by their scaled-down copies (up to 5 times).
  - Sum the results of all frames.
  - Calculate the sum in frequency ranges for both genders.
  - Classify as male or female based on the higher score.

### Data Loading:
- File Format: `.wav`
- Gender identification based on file name.

### Data Processing:
- Use `librosa` for loading audio data.
- Process each audio file through HPS.
- Record actual and predicted genders.

### Reporting:
- Enabling report option triggers the display of the confusion matrix and classification report.

## Algorithm Performance

### Confusion Matrix and Classification Report:
The performance of the gender recognition algorithm is quantitatively assessed using a confusion matrix and a classification report. These metrics provide insights into the precision, recall, and overall accuracy of the algorithm in classifying male and female voices.

```plaintext
Classification Report:
              precision    recall  f1-score   support

           K       0.92      0.98      0.95        45
           M       0.98      0.91      0.94        44

    accuracy                           0.94        89
   macro avg       0.95      0.94      0.94        89
weighted avg       0.95      0.94      0.94        89 
```

### Overall Classification Accuracy:
- **94%**

## Final Remarks

- The algorithm effectively recognizes gender from voice with an overall accuracy of 94%.
- High precision and recall values indicate a strong ability to differentiate between male and female voices.
- The confusion matrix and classification report provide detailed insights into the algorithm's effectiveness in various scenarios.
