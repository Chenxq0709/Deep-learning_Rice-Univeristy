from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from tensorflow.keras import layers, models, callbacks, backend
import matplotlib.pyplot as plt
import re
from skimage import io
from itertools import product

def load_modified_cifar10():
    class_regex = re.compile(r'.*\\(\d)\\.*')
    train_data = io.imread_collection('CIFAR10\\Train\\*\\*.png')
    test_data = io.imread_collection('CIFAR10\\Test\\*\\*.png')

    train_labels = np.array([int(class_regex.match(path).group(1)) for path in train_data.files])[:, None]
    test_labels = np.array([int(class_regex.match(path).group(1)) for path in test_data.files])[:, None]
    # To verify that the dataset looks correct
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # plt.figure(figsize=(10, 4))
    # for i,j in product(range(10), range(1)):
    #     plt.subplot(2, 5, i+j + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_data[i*1000+j],cmap='gray', vmin=0, vmax=255)
    #     plt.xlabel(class_names[train_labels[i*1000+j][0]])
    # plt.show()

    train_data_processed = np.stack(train_data).astype(float) / 255
    train_data_processed = train_data_processed.reshape((-1, 28, 28, 1))
    test_data_processed = np.stack(test_data).astype(float) / 255
    test_data_processed = test_data_processed.reshape((-1, 28, 28, 1))
    return train_data_processed, train_labels, test_data_processed, test_labels


# 1.1 CIFAR10 Dataset
train_images, train_labels, test_images, test_labels = load_modified_cifar10()


# 1.2 Train LeNet5 on CIFAR10
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
earlystop = callbacks.EarlyStopping( monitor='val_acc', min_delta=0.0001, patience=1)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
Adam = model.fit(train_images, train_labels, batch_size= 50,  epochs=100,   callbacks=[earlystop], validation_data=(test_images, test_labels))

model.save('Adam')

plt.plot(Adam.history['loss'], label='train')
plt.plot(Adam.history['val_loss'], label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(Adam.history['acc'], label='train')
plt.plot(Adam.history['val_acc'], label= 'test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

Adam_test_loss, Adam_test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('optimizer=Adam, test accuracy=', Adam_test_acc)
# # 1.3 Visualize the Trained Network

w1 = model.layers[0].get_weights()[0]
w1_std = np.std(w1.reshape((-1, w1.shape[-1])), axis = 0)
best_indices = list(np.argsort(w1_std)[:: -1][: 4])

vmax = np.abs(w1).max()
vmin = -vmax
plt.figure()
for i, j in product(range(4), range(8)):
    k = i * 8 + j
    plt.subplot(4, 8, k + 1)
    plt.pcolormesh(np.flipud(w1[:, :, 0, k]), vmin = vmin, vmax = vmax, cmap = 'gray')
    plt.axis('off')

plt.show()
gridspec = plt.GridSpec(4, 8)

for i1, j1 in product(range(2), range(2)):
    k = i1 * 2 + j1
    plt.subplot(gridspec[i1 * 2: i1 * 2 + 2, j1 * 4: j1 * 4 + 2])
    img = test_images[200 * k + 10, :, :, 0]
    plt.pcolormesh(np.flipud(img), vmin = 0, vmax = 1, cmap = 'gray')
    plt.axis('off')
    get_pool1_layer_output = backend.function([model.layers[0].input], [model.layers[1].output])
    h1 = get_pool1_layer_output([img.reshape((1, 28, 28, 1))])[0]
    for i2, j2 in product(range(2), range(2)):
        k2 = i2 * 2 + j2
        plt.subplot(gridspec[i1 * 2 + i2, j1 * 4 + 2 + j2])
        h = h1[0, :, :, best_indices[k2]]
        plt.pcolormesh(np.flipud(h), cmap = 'gray')
        plt.axis('off')

plt.show()