from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers\
import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model")
args = vars(ap.parse_args())

INITIAL_LEARING_RATE = 1e-4
EPOCHS = 50
batchSize = 32

print("\033[2;32;m [INFO] loading images.... \n")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels =1057716\Miniconda3\envs\tensorflow\python.exe D:/pycharm/mask_detect/mask-model.py
 [INFO] loading images....

WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
2020-08-29 17:21:59.629647: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-08-29 17:21:59.805512: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x23f5e806b90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-29 17:21:59.805829: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
 [INFO] compiling model...
 [INFO] training head...
Epoch 1/50
WARNING:tensorflow:AutoGraph could not transform <function Model.make_train_function.<locals>.train_function at 0x00000242DB649438> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'arguments' object has no attribute 'posonlyargs'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2020-08-29 17:22:11.829124: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 154140672 exceeds 10% of free system memory.
2020-08-29 17:22:11.952918: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 156905472 exceeds 10% of free system memory.
  1/310 [..............................] - ETA: 0s - loss: 0.6771 - accuracy: 0.68752020-08-29 17:22:12.963342: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 154140672 exceeds 10% of free system memory.
2020-08-29 17:22:13.056729: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 156905472 exceeds 10% of free system memory.
  2/310 [..............................] - ETA: 2:50 - loss: 0.6075 - accuracy: 0.71882020-08-29 17:22:14.209640: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 154140672 exceeds 10% of free system memory.
310/310 [==============================] - ETA: 0s - loss: 0.1040 - accuracy: 0.9624WARNING:tensorflow:AutoGraph could not transform <function Model.make_test_function.<locals>.test_function at 0x00000242DB973048> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'arguments' object has no attribute 'posonlyargs'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 77 batches). You may need to use the repeat() function when building your dataset.
310/310 [==============================] - 417s 1s/step - loss: 0.1040 - accuracy: 0.9624 - val_loss: 0.0181 - val_accuracy: 0.9944
Epoch 2/50
310/310 [==============================] - 332s 1s/step - loss: 0.0248 - accuracy: 0.9931
Epoch 3/50
310/310 [==============================] - 339s 1s/step - loss: 0.0151 - accuracy: 0.9950
Epoch 4/50
310/310 [==============================] - 341s 1s/step - loss: 0.0120 - accuracy: 0.9963
Epoch 5/50
310/310 [==============================] - 335s 1s/step - loss: 0.0085 - accuracy: 0.9969
Epoch 6/50
310/310 [==============================] - 337s 1s/step - loss: 0.0075 - accuracy: 0.9972
Epoch 7/50
310/310 [==============================] - 363s 1s/step - loss: 0.0080 - accuracy: 0.9971
Epoch 8/50
310/310 [==============================] - 363s 1s/step - loss: 0.0080 - accuracy: 0.9978
Epoch 9/50
310/310 [==============================] - 362s 1s/step - loss: 0.0061 - accuracy: 0.9984
Epoch 10/50
310/310 [==============================] - 365s 1s/step - loss: 0.0048 - accuracy: 0.9987
Epoch 11/50
310/310 [==============================] - 363s 1s/step - loss: 0.0063 - accuracy: 0.9980
Epoch 12/50
310/310 [==============================] - 354s 1s/step - loss: 0.0041 - accuracy: 0.9988
Epoch 13/50
310/310 [==============================] - 336s 1s/step - loss: 0.0056 - accuracy: 0.9987
Epoch 14/50
310/310 [==============================] - 364s 1s/step - loss: 0.0039 - accuracy: 0.9988
Epoch 15/50
310/310 [==============================] - 357s 1s/step - loss: 0.0046 - accuracy: 0.9985
Epoch 16/50
310/310 [==============================] - 357s 1s/step - loss: 0.0045 - accuracy: 0.9985
Epoch 17/50
310/310 [==============================] - 343s 1s/step - loss: 0.0051 - accuracy: 0.9983
Epoch 18/50
310/310 [==============================] - 346s 1s/step - loss: 0.0037 - accuracy: 0.9985
Epoch 19/50
310/310 [==============================] - 343s 1s/step - loss: 0.0029 - accuracy: 0.9989
Epoch 20/50
310/310 [==============================] - 350s 1s/step - loss: 0.0038 - accuracy: 0.9991
Epoch 21/50
310/310 [==============================] - 360s 1s/step - loss: 0.0044 - accuracy: 0.9983
Epoch 22/50
310/310 [==============================] - 345s 1s/step - loss: 0.0030 - accuracy: 0.9989
Epoch 23/50
310/310 [==============================] - 327s 1s/step - loss: 0.0045 - accuracy: 0.9988
Epoch 24/50
310/310 [==============================] - 327s 1s/step - loss: 0.0043 - accuracy: 0.9987
Epoch 25/50
310/310 [==============================] - 328s 1s/step - loss: 0.0022 - accuracy: 0.9995
Epoch 26/50
310/310 [==============================] - 329s 1s/step - loss: 0.0018 - accuracy: 0.9995
Epoch 27/50
310/310 [==============================] - 326s 1s/step - loss: 0.0022 - accuracy: 0.9994
Epoch 28/50
310/310 [==============================] - 325s 1s/step - loss: 0.0031 - accuracy: 0.9990
Epoch 29/50
310/310 [==============================] - 334s 1s/step - loss: 0.0014 - accuracy: 0.9997
Epoch 30/50
310/310 [==============================] - 357s 1s/step - loss: 0.0036 - accuracy: 0.9992
Epoch 31/50
310/310 [==============================] - 343s 1s/step - loss: 0.0020 - accuracy: 0.9993
Epoch 32/50
310/310 [==============================] - 325s 1s/step - loss: 0.0022 - accuracy: 0.9993
Epoch 33/50
310/310 [==============================] - 325s 1s/step - loss: 0.0015 - accuracy: 0.9995
Epoch 34/50
310/310 [==============================] - 330s 1s/step - loss: 0.0018 - accuracy: 0.9995
Epoch 35/50
310/310 [==============================] - 342s 1s/step - loss: 0.0011 - accuracy: 0.9995
Epoch 36/50
310/310 [==============================] - 334s 1s/step - loss: 0.0019 - accuracy: 0.9993
Epoch 37/50
310/310 [==============================] - 323s 1s/step - loss: 0.0014 - accuracy: 0.9994
Epoch 38/50
310/310 [==============================] - 323s 1s/step - loss: 9.4892e-04 - accuracy: 0.9999
Epoch 39/50
310/310 [==============================] - 321s 1s/step - loss: 0.0022 - accuracy: 0.9995
Epoch 40/50
310/310 [==============================] - 344s 1s/step - loss: 0.0029 - accuracy: 0.9989
Epoch 41/50
310/310 [==============================] - 347s 1s/step - loss: 0.0033 - accuracy: 0.9990
Epoch 42/50
310/310 [==============================] - 337s 1s/step - loss: 0.0015 - accuracy: 0.9995
Epoch 43/50
310/310 [==============================] - 325s 1s/step - loss: 0.0012 - accuracy: 0.9998
Epoch 44/50
310/310 [==============================] - 327s 1s/step - loss: 0.0020 - accuracy: 0.9993
Epoch 45/50
310/310 [==============================] - 329s 1s/step - loss: 0.0012 - accuracy: 0.9996
Epoch 46/50
310/310 [==============================] - 336s 1s/step - loss: 0.0015 - accuracy: 0.9995
Epoch 47/50
 to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

print("\033[2;32;m [INFO] compiling model...")
opt = Adam(lr=INITIAL_LEARING_RATE, decay=INITIAL_LEARING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the head of the network
print("\033[2;32;m [INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batchSize),
    steps_per_epoch=len(trainX) // batchSize,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batchSize,
    epochs=EPOCHS)

print("\033[2;32;m [INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=batchSize)

predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
print("\033[2;37;m [INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])