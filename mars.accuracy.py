
import matplotlib.pyplot as plt

# models & accuracies
models = ["SVM", "CNN"]
accuracies = [0.49, 0.92]  

# plot
plt.figure(figsize=(6, 4))
plt.bar(models, accuracies, color=["skyblue", "pink"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison: SVM vs CNN")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.show()
