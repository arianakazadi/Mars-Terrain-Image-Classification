# Mars-Terrain-Image-Classification

This project compares classical machine learning and deep learning approaches for classifying Martian terrain using labeled satellite imagery, including SVM and convolutional neural network models.

Repository Structure:
  LICENSE - MIT open-source license
  README.md - project overview and usage description
  sample.images.mars.pdf - Sample subset of 4 images per class (8)
    **Dataset:** Full Mars terrain dataset (6,000+ images, 8 classes) available at (https://www.kaggle.com/datasets/aumthaker/mars-terrain-images?resource=download)
  mars.svm.py - Python script containing preprocessing, SVM model training, and evaluation
  mars.cnn.py - Python script containing preprocessing, CNN model training, and evaluation
  mars.accuracy - python script for the comparison graph
  mars.plots.pdf - output visualizations including confusion matrices and model comparison figures
  requirements.txt - Python dependencies for reproducibility
