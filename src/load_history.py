import pickle
import matplotlib.pyplot as plt
# Load the history from a pickle file
# This script loads the training history of a model from a pickle file and visualizes the accuracy and loss over epochs.
# Load the history
with open('history_rps.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# Print or analyze the history
print(loaded_history)

# Plotting the training and validation accuracy and loss
plt.figure(figsize=(12, 6))
plt.plot(loaded_history['accuracy'], label='Training Accuracy')
plt.plot(loaded_history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(loaded_history['loss'], label='Training Loss')
plt.plot(loaded_history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = loaded_history.history['accuracy']
val_acc = loaded_history.history['val_accuracy']
loss = loaded_history.history['loss']
val_loss = loaded_history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()