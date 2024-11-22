# MNIST Digit Classifier

An implementation of a digit classification system using the MNIST dataset. The project provides options to train a new model, use an existing one, and test the model with custom images. It includes features such as TensorBoard to monitor training performance and early stopping for efficient model training.

## Features

- **Model Training**: Build and train a CNN for digit classification using TensorFlow/Keras.
- **Model Saving and Loading**: Save trained models for reuse, avoiding the need for retraining every time.
- **TensorBoard Integration**: Monitor training logs, metrics, and performance.
- **Early Stopping**: Stop training if validation loss does not improve, saving time and resources.

## Requirements

Ensure you have the following installed:

- Python 3.8+
- TensorFlow
- Matplotlib

Install required packages with:

```bash 
pip install -r requirements.txt
```
## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. Run the program:

   ```bash
   python mnist_classifier.py
   ```

3. Choose an option:
   - Train a new model.
   - Use an existing model.

## Project Structure

```plaintext
├── main.py             # Main script
├── mnist_digit_classifier.h5  # Pre-trained model (if available)
├── logs/               # TensorBoard logs
├── custom_images/      # Folder for custom test images
└── README.md           # Project documentation
```

## Example Output

- **Training Output**:
  Displays loss, accuracy, and TensorBoard log directory.

## TensorBoard

To monitor training logs:

```bash
tensorboard --logdir=logs/fit
```

Open the displayed URL in your browser.

## Contribution

Feel free to fork this repository, make changes, and submit pull requests. Contributions are welcome!