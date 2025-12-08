import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATA_DIR = "planesnet"   #Name of the file containing all .png

images = []
labels = []

for fname in os.listdir(DATA_DIR):
    if fname.lower().endswith(".png"):
        
        #The name format of images starts with a 0 if there is no-plane or 1 if there is a plane:
        label = int(fname[0])
        fpath = os.path.join(DATA_DIR, fname)

        img = load_img(fpath, target_size=(20, 20))
        #We normalize the values (20x20x3) for the neural network:
        img = img_to_array(img) / 255.0

        images.append(img)
        labels.append(label)

images = np.array(images)         # (N, 20, 20, 3)
labels = np.array(labels)         # (N,)
#Conversion to one-hot vector:
labels_cat = to_categorical(labels, num_classes=2) 


#Splitting data into training set, validation set and testing set
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_cat, test_size=0.2, random_state=42, stratify=labels
)


#Construction of the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(20, 20, 3)))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Conv2D(128, (3, 3), activation="relu"))


model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

#Training
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32
)

#Final test
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

#Visualisation of results:

#20 random images in the dataset
idxs = np.random.choice(len(images), 20, replace=False)
plt.figure(figsize=(10,5))
for i, idx in enumerate(idxs):
    plt.subplot(4,5,i+1)
    plt.imshow(images[idx])
    plt.axis("off")
    plt.title("plane" if labels[idx] == 1 else "no plane")
plt.tight_layout()
plt.show()

#Evolution of accuracy
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.title("Accuracy evolution")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

#20 correct predictions
preds = model.predict(X_test)
pred_classes = preds.argmax(axis=1)
true_classes = y_test.argmax(axis=1)

good = np.where(pred_classes == true_classes)[0]
good_idx = np.random.choice(good, 20, replace=False)

plt.figure(figsize=(10,5))
for i, idx in enumerate(good_idx):
    plt.subplot(4,5,i+1)
    plt.imshow(X_test[idx])
    plt.axis("off")
    label = "plane" if true_classes[idx] == 1 else "no plane"
    plt.title(label)
plt.tight_layout()
plt.show()

#20 false predictions
bad = np.where(pred_classes != true_classes)[0]
bad = bad[:20]

plt.figure(figsize=(10,5))
for i, idx in enumerate(bad):
    plt.subplot(4,5,i+1)
    plt.imshow(X_test[idx])
    plt.axis("off")
    t = "plane" if true_classes[idx] == 1 else "no plane"
    p = "plane" if pred_classes[idx] == 1 else "no plane"
    plt.title(f"T:{t} / P:{p}")
plt.tight_layout()
plt.show()

#Confusion matrix
cm = confusion_matrix(true_classes, pred_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=["no plane", "plane"])
disp.plot(cmap="Blues") 
plt.show()
