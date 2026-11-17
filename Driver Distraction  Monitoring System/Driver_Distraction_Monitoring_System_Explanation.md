# Driver Distraction Monitoring System: How It Works

This project is a computer vision system designed to detect whether a driver is distracted and, if so, what kind of distraction is happening. It uses a custom ResNet-50 deep learning model trained to classify driver behavior from video frames.

The project has three main parts:

1. Data preparation and model training logic in the notebook file `Driver_Distraction_Monitoring_System.ipynb`
2. The custom neural network architecture in `resnet_50.py`
3. Real-time video inference and user interface in `inference.py`

---

## 1. What problem is the system solving?

The model classifies driver activity into 10 classes:

- `c0` — safe driving
- `c1` — texting - right
- `c2` — talking on the phone - right
- `c3` — texting - left
- `c4` — talking on the phone - left
- `c5` — operating the radio
- `c6` — drinking
- `c7` — reaching behind
- `c8` — hair and makeup
- `c9` — talking to passenger

This is a classic image classification problem where each frame of a driving video is treated as one input sample and the model predicts the likely driver action.

---

## 2. Dataset and training idea

The notebook uses the Kaggle dataset named `state-farm-distracted-driver-detection`.

The data is arranged in folders such as:

- `train/c0/...`
- `train/c1/...`
- ...
- `train/c9/...`

Each class directory contains images associated with a single driver distraction behavior.

The notebook performs the following:

- uploads the Kaggle API token
- downloads the dataset from Kaggle
- unzips it into the working directory
- randomly samples images for visualization
- splits the data into train/validation sets
- prepares data loaders for preprocessing

The key logic is:

- `paths.list_images(...)` gets all image file paths
- the images are shuffled and selected into train/validation buckets
- each image is resized before being passed to the model

This is essential because deep learning models expect a consistent image size, and a real-world dataset often contains images of different dimensions.

---

## 3. Preprocessing pipeline

The project uses custom preprocessing classes in the notebook:

### `SimpleDatasetLoader`

This class loads image files and attaches labels based on the folder name.

Example logic:

- read an image with `cv2.imread(imagePath)`
- label = the parent folder name, such as `c3`
- apply preprocessors (if any)
- append image to `data`
- append label to `labels`
- return `np.array(data)` and `np.array(labels)`

This is a standard way to convert raw image files into training arrays for a neural network.

### `SimplePreprocessor`

This class resizes every image to a fixed shape, such as `(width, height)`, using OpenCV:

```python
cv2.resize(image, (self.width, self.height), interpolation=self.inter)
```

This ensures that all images have the same dimension before entering the CNN.

### `ImageToArrayPreprocessor`

This converts the NumPy/OpenCV image array into a Keras-compatible array using:

```python
img_to_array(image, data_format=self.dataFormat)
```

This step is necessary because Keras expects image tensors in a specific shape.

---

## 4. The deep learning model: custom ResNet-50

The model architecture is implemented in `resnet_50.py`.

This is not the official large ResNet-50 from TensorFlow/Keras; it is a custom implementation inspired by the ResNet architecture, made to fit the distraction classification task.

### Why ResNet?

ResNet solves a major problem in very deep networks: vanishing gradients.

When neural networks become too deep, training gets harder because gradients become very small and important information can be lost. ResNet solves this by using residual connections (skip connections).

Instead of trying to learn the entire transformation from scratch, the network learns:

- the main path output
- the original input shortcut path
- then adds them together

This makes deep networks easier to train.

---

## 5. ResNet building blocks

### `identity_block`

This function creates a residual block where the input and output dimensions remain the same.

The structure is:

1. 1x1 convolution
2. Batch normalization
3. ReLU
4. 3x3 convolution
5. Batch normalization
6. ReLU
7. 1x1 convolution
8. Batch normalization
9. Add original input (`X_skip_connection`)
10. ReLU

This is the core idea of ResNet: the shortcut path bypasses the stacked layers and adds the original data back.

```python
X = Add()([X, X_skip_connection])
X = Activation('relu')(X)
```

This is what makes the model trainable even with many layers.

### `convolutional_block`

This is slightly different from an identity block because the input and output shapes may change.

It is used when the number of channels or the spatial size needs to change.

The function:

- uses a 1x1 convolution to reduce or adjust dimensions
- applies the usual 3x3 convolution
- uses another 1x1 convolution to generate the final output
- creates a skip connection through a matching 1x1 convolution as well
- adds both paths together

This allows the network to downsample while preserving the information flow.

---

## 6. Model architecture in detail

The `ResNet50` function defines the entire model.

### Input Layer

```python
X_input = Input(input_shape)
```

The model expects an image of shape:

```python
(128, 128, 3)
```

That means:

- 128 pixels wide
- 128 pixels tall
- 3 color channels (RGB)

### Zero Padding

```python
X = ZeroPadding2D((3, 3))(X_input)
```

This adds extra border pixels around the input so that the convolution operations can work cleanly without losing image edge information.

### Initial Convolution and Pooling

```python
X = Conv2D(64, (7, 7), strides=(2, 2), ...)(X)
X = BatchNormalization(...)(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)
```

This stage extracts low-level features such as edges, gradients, and textures.

### Stages 2 to 5

These are the deep residual stages:

- Stage 2: 3 residual blocks
- Stage 3: 4 residual blocks
- Stage 4: 6 residual blocks
- Stage 5: 3 residual blocks

Each stage uses a combination of convolutional blocks and identity blocks.

This gradually learns more complex patterns in the image, from simple visual features to high-level distraction behaviors.

### Average Pooling

```python
X = AveragePooling2D((2, 2), name='avg_pool')(X)
```

This reduces the feature map size while keeping the most important information.

### Flatten + Dense Layers

```python
X = Flatten()(X)
X = Dense(512, activation='relu')(...)
X = Dense(256, activation='relu')(...)
X = Dense(128, activation='relu')(...)
X = Dense(64, activation='relu')(...)
X = Dense(classes, activation='softmax', name='fc10')(...)
```

These dense layers interpret the learned image features and convert them into 10 class scores.

The final softmax layer outputs probabilities for the 10 distraction classes:

```python
Dense(classes, activation='softmax')
```

The predicted class is the label with the highest probability.

---

## 7. Real-time monitoring in `inference.py`

This file is the operational part of the project. It loads the trained weights and processes a video file frame by frame.

### Model loading

```python
model = ResNet50(input_shape=(128,128,3), classes=10)
model.load_weights('restnet_50.weights.h5')
```

This creates the network architecture and loads weights previously trained and saved in the file `restnet_50.weights.h5`.

The model is then ready to classify frames from a video.

---

## 8. GUI flow with Tkinter

The system uses Tkinter to create a windows-based desktop interface.

### Main components

- `Choose File` button: opens a file dialog for video selection
- `Start` button: starts processing the video
- `Stop` button: stops monitoring
- `video_label`: shows the video frame with prediction overlay

### File selection

```python
def choose_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
```

The user picks a `.mp4` or `.avi` file, and the chosen path is stored in `video_path`.

---

## 9. Video processing loop

### `start_monitoring()`

```python
def start_monitoring():
    global running, cap
    if video_path:
        running = True
        cap = cv2.VideoCapture(video_path)
        show_frame()
```

This opens the selected video using OpenCV and starts the frame-processing loop.

### `show_frame()`

This is the heart of the project.

```python
ret, frame = cap.read()
```

It reads one frame from the video. If the frame is missing, the monitor stops.

Then it resizes the frame for display:

```python
frame = cv2.resize(frame, (display_width, display_height))
```

Then it prepares the frame for the model:

```python
data = cv2.resize(frame,(128,128))
data = data.astype("float") / 255.0
data = np.expand_dims(data, axis=0)
```

This does three things:

- resizes to model input size `(128,128)`
- normalizes pixel values from `0..255` to `0..1`
- adds a batch dimension so Keras can process it

This conversion is essential because the model expects a 4D tensor of shape `(batch, height, width, channels)`.

---

## 10. Prediction step

```python
prediction = model.predict(data)
```

This returns a vector of 10 class probabilities.

```python
predicted_class = Name[prediction.argmax()]
```

- `prediction.argmax()` returns the index of the highest probability
- this index is mapped to a label list called `Name`

Example label list:

```python
Name = [
    "safe driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger"
]
```

So if the model predicts `index 6`, it will label the frame as `drinking`.

---

## 11. Overlaying result on the video

Once the prediction is made, the system overlays the label on the video frame:

```python
cv2.putText(frame, f'Driver Distraction Monitoring: {predicted_class}', (30,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
```

This creates a real-time display that says something like:

- `Driver Distraction Monitoring: texting - right`
- `Driver Distraction Monitoring: safe driving`

The text is drawn in red and placed at the top-left of the video.

---

## 12. Displaying the frame in the GUI

The processed frame is converted from BGR to RGB for Tkinter display:

```python
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame)
imgtk = ImageTk.PhotoImage(image=img)
```

Then the frame is set onto the UI label:

```python
video_label.imgtk = imgtk
video_label.configure(image=imgtk)
```

This updates the preview window continuously.

---

## 13. Continuous loop

To process the video continuously, the code schedules itself:

```python
root.after(10, show_frame)
```

This means after every 10 milliseconds, the next frame is fetched and processed.

This creates the effect of real-time monitoring.

---

## 14. Stop logic

```python
def stop_monitoring():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    video_label.config(image="")
    sys.exit(1)
```

This shuts down the video capture and closes the monitoring loop.

---

## 15. End-to-end workflow of the full system

Here is the complete flow of the project:

1. The user selects a driving video in the GUI.
2. The app opens the video with OpenCV.
3. Each frame is resized to `(128,128,3)` and normalized.
4. The ResNet-50 model predicts one of 10 distraction states.
5. The predicted class is displayed as text on the video stream.
6. The processed frame is shown in the Tkinter window.
7. This continues until the user stops the monitoring.

---

## 16. Overall understanding of the design

This project combines three important technologies:

- OpenCV for reading and processing video frames
- TensorFlow/Keras for the deep learning model
- Tkinter for the desktop user interface

The design is straightforward and effective for a student or prototype project:

- a custom CNN/ResNet system learns the pattern of distracted driving
- it runs on video input
- it predicts the current behavior in real time
- it displays the prediction for the user

---

## 17. Important technical observations

A few important notes:

- The project is a prototype and not a production-grade safety system.
- It relies heavily on pretrained weights saved in `restnet_50.weights.h5`.
- The file name is slightly inconsistent (`restnet_50` vs `resnet_50`), so weight loading must match exactly.
- The app is designed for video-based classification and is best suited for offline or controlled evaluation rather than a siren-like real-world deployment system.

---

## 18. Summary

The system works by:

- collecting images of driver behavior,
- training a ResNet-inspired CNN to classify them,
- loading saved weights,
- processing each video frame,
- predicting the current distraction state,
- displaying the result directly on the video feed.

In simple terms, it is a real-time driver behavior recognition system that watches a driver and tells whether they are driving safely or distracted, and what kind of distraction is occurring.

---

## 19. Short code logic summary

```python
frame = cv2.imread(...) or cap.read()
frame = cv2.resize(frame, (128, 128))
frame = frame.astype('float') / 255.0
frame = np.expand_dims(frame, axis=0)
probabilities = model.predict(frame)
class_index = probabilities.argmax()
label = class_names[class_index]
print(label)
```

This is the essence of the whole system.
