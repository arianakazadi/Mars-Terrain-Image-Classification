
from PIL import Image
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



DATA_DIR = "mars_images"  
IMAGE_SIZE = (128, 128)   

#class names
classes = [c for c in os.listdir(DATA_DIR) if not c.startswith(".")]
print("Classes found:", classes)

#loop
for label in classes:
    class_path = os.path.join(DATA_DIR, label)
   
    img_name = os.listdir(class_path)[0]
    img_path = os.path.join(class_path, img_name)

    #resize image
    img = Image.open(img_path)
    img = img.resize(IMAGE_SIZE)

    #display image
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")
    plt.show()

   print(f"{label}: image size = {img.size}")


X = []
y = []

for label in classes:
    class_path = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).resize(IMAGE_SIZE)
        img_array = np.array(img)         
        X.append(img_array.flatten())     
        y.append(label)                    

#conversion
X = np.array(X)
y = np.array(y)

print("Feature array shape:", X.shape)
print("Labels array shape:", y.shape)



le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded labels:", np.unique(y_encoded))



X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

#predict
y_pred = svm_model.predict(X_test)

#evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

#cm
cm_svm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm_svm)
plt.colorbar()
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(le.classes_)), le.classes_, rotation=45)
plt.yticks(np.arange(len(le.classes_)), le.classes_)
plt.tight_layout()
plt.show()

