import matplotlib.pyplot as plt
from data.mnist_data import load_data
from model.cnn_model import create_model

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Create model
    model = create_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

    # Visualize some predictions
    predictions = model.predict(x_test)
    plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predictions[0].argmax()}')
    plt.show()

if __name__ == "__main__":
    main()
