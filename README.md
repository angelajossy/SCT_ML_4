# Hand Gesture Recognition Model

This project is a deep learning model developed to accurately identify and classify 10 different hand gestures. The model is built with TensorFlow/Keras and trained on the **Hand Gesture Recognition Database** from Kaggle, achieving 99.95% validation accuracy.

This repository contains the code and documentation for the first phase of this project: **training and validating the core recognition model**.

 \#\# Dataset

  * **Source:** [Hand Gesture Recognition Database (LeapGestRecog)](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
  * **Contents:** The dataset consists of 20,000 images of 10 different hand gestures (10 subjects, 2,000 images per gesture).
  * **Gestures:**
      * `01_palm`
      * `02_l`
      * `03_fist`
      * `04_fist_moved`
      * `05_thumb`
      * `06_index`
      * `07_ok`
      * `08_palm_moved`
      * `09_c`
      * `10_down`

-----

## Project Methodology

The project was developed in a Kaggle Notebook, leveraging a T4 GPU for accelerated training.

### 1\. Data Preparation

The original dataset is sorted by subject (e.g., `.../00/01_palm/img.png`). To make it compatible with Keras, a custom script was used to:

1.  **Reorganize:** Copy all images into a new directory structure sorted by class (e.g., `.../processed/01_palm/img.png`).
2.  **Subset:** Create a subset of 10,000 images (1,000 per class) to create a balanced and sufficiently large dataset for training.

### 2\. Data Loading and Pre-processing

  * **Loading:** The `tf.keras.utils.image_dataset_from_directory` function was used to load the 10,000 images from the reorganized directory.
  * **Splitting:** The data was split into **8,000 training images** (80%) and **2,000 validation images** (20%).
  * **Pre-processing:**
      * **Image Size:** All images were resized to $96 \times 96$ pixels.
      * **Color Mode:** Loaded as `grayscale` (1 color channel).
      * **Batching:** Grouped into batches of 32.
      * **Optimization:** The data pipeline was optimized using `.cache()`, `.shuffle()`, and `.prefetch()` for maximum performance on the GPU.

### 3\. Model Architecture

A Convolutional Neural Network (CNN) was built using the Keras Sequential API. The architecture is as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
rescaling (Rescaling)        (None, 96, 96, 1)         0         
                                                                 
conv2d (Conv2D)              (None, 96, 96, 16)        160       
                                                                 
max_pooling2d (MaxPooling2D) (None, 48, 48, 16)        0         
                                                                 
conv2d_1 (Conv2D)            (None, 48, 48, 32)        4640      
                                                                 
max_pooling2d_1 (MaxPooling2 (None, 24, 24, 32)        0         
                                                                 
flatten (Flatten)            (None, 18432)             0         
                                                                 
dense (Dense)                (None, 128)               2359424   
                                                                 
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 2,365,514
Trainable params: 2,365,514
Non-trainable params: 0
_________________________________________________________________
```

### 4\. Training

  * **Compiler:** The model was compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function.
  * **Training:** The model was trained for 5 epochs. Thanks to the T4 GPU, each epoch completed in just a few seconds.

-----

## Results

The model achieved excellent performance, reaching 100% training accuracy and 99.95% validation accuracy, indicating a near-perfect fit to the dataset.

#### Training Log:

```
Starting model training...
Epoch 1/5
250/250 ━━━━━━━━ 10s 12ms/step - accuracy: 0.7581 - loss: 0.7922 - val_accuracy: 0.9960 - val_loss: 0.0189
Epoch 2/5
250/250 ━━━━━━━━ 1s 6ms/step - accuracy: 0.9985 - loss: 0.0087 - val_accuracy: 0.9990 - val_loss: 0.0038
Epoch 3/5
250/250 ━━━━━━━━ 1s 5ms/step - accuracy: 0.9999 - loss: 9.3199e-04 - val_accuracy: 0.9995 - val_loss: 0.0035
Epoch 4/5
250/250 ━━━━━━━━ 1s 5ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 0.9995 - val_loss: 0.0011
Epoch 5/5
250/250 ━━━━━━━━ 1s 5ms/step - accuracy: 1.0000 - loss: 9.3474e-05 - val_accuracy: 0.9995 - val_loss: 0.0014
Training finished.
```

#### Performance Graphs:

 As seen in the graphs, the model learned extremely quickly, with the loss dropping to near-zero and accuracy hitting 100% by the second epoch.

