import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
import datetime

# Función para construir y entrenar el modelo
def train_and_save_model(train_images, train_labels, test_images, test_labels, model_path):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        train_images, 
        train_labels, 
        epochs=20, 
        validation_data=(test_images, test_labels), 
        callbacks=[early_stopping, tensorboard_callback]
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")
    return model

# Función para cargar un modelo preentrenado
def load_existing_model(model_path):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        return tf.keras.models.load_model(model_path)
    else:
        print("Model file not found. Please train a new model first.")
        return None

# Cargar el dataset MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocesamiento: normalizar las imágenes
train_images = train_images / 255.0
test_images = test_images / 255.0

# Remodelar las imágenes para que sean compatibles con el modelo
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Ruta para guardar o cargar el modelo
model_path = "mnist_digit_classifier.h5"

# Menú interactivo
print("Choose an option:")
print("1. Train a new model")
print("2. Load an existing model")
choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    model = train_and_save_model(train_images, train_labels, test_images, test_labels, model_path)
elif choice == "2":
    model = load_existing_model(model_path)
    if model is None:
        print("No model available. Training a new model...")
        model = train_and_save_model(train_images, train_labels, test_images, test_labels, model_path)
else:
    print("Invalid choice. Exiting.")
    exit()

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Mostrar la predicción de la primera imagen de prueba solo en la terminal
predictions = model.predict(test_images)
print(f"First test image predicted label: {predictions[0].argmax()}")

# Preguntar si se desea probar con una imagen personalizada
while True:
    test_choice = input("Do you want to test with a custom image? (yes/no): ").strip().lower()
    if test_choice == "yes":
        image_path = input("Enter the path to the image: ").strip()
        try:
            from PIL import Image
            img = Image.open(image_path).convert("L")  # Convertir a escala de grises
            img = img.resize((28, 28))  # Redimensionar a 28x28
            img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalizar
            img_array = img_array.reshape((1, 28, 28, 1))  # Ajustar la forma para el modelo
            
            prediction = model.predict(img_array)
            print(f"Predicted label: {prediction.argmax()}")
            
            # Mostrar la imagen procesada
            plt.imshow(img_array.reshape(28, 28), cmap=plt.cm.binary)
            plt.title(f"Predicted Label: {prediction.argmax()}")
            plt.show()
        except Exception as e:
            print(f"Error loading or processing the image: {e}")
    elif test_choice == "no":
        print("Exiting.")
        break
    else:
        print("Invalid input. Please type 'yes' or 'no'.")
