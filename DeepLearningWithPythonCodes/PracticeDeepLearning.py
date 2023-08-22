from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
# from typing import Any

# import numpy as np
# x = np.array(12)
# x.ndim
# print(x.ndim)

# import numpy as np
# x = np.array([12, 3, 6, 14])
# x.ndim
# print(x.ndim)

# import numpy as np
# x = np.array([[5, 78, 2, 34, 0],
# [6, 79, 3, 35, 1],
# [7, 80, 4, 36, 2]])
# x.ndim
# print(x.ndim)

# import numpy as np
# x = np.array([[[5, 78, 2, 34, 0],
# [6, 79, 3, 35, 1],
# [7, 80, 4, 36, 2]],
# [[5, 78, 2, 34, 0],
# [6, 79, 3, 35, 1],
# [7, 80, 4, 36, 2]],
# [[5, 78, 2, 34, 0],
# [6, 79, 3, 35, 1],
# [7, 80, 4, 36, 2]]])
# x.ndim
# print(x.ndim)

# from keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.ndim)


# from keras.datasets import mnist
# from keras import models
# from keras import layers
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,60000)))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# digit = train_images[4]
# plt.imshow(digit , cmap=plt.cm.binary)
# plt.show()
# my_slice = train_images[10:100]
# print(my_slice.shape)
# print('test_acc:', test_acc)


# from keras.datasets import mnist
# from keras import models
# from keras import layers
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# my_slice = train_images[10:100]
# print('test_acc:', test_acc)
# print(my_slice.shape)

# batch = train_images[:128]-----size 129
#
# batch1 = train_images[128:256]-----size 256
#
# batch2 = train_images[128 * n:128 * (n + 1)]------n size of a batch

# def naive_relu(x):
#     assert len(x.shape) == 2
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] = max(x[i, j], 0)
#     return x

# def naive_add(x, y):
#     assert len(x.shape) == 2
#     assert x.shape == y.shape
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] += y[i, j]
#     return x

# import numpy as np
# x = 1
# y = 2
# z = x + y
# z = np.maximum(z, 0.)
# print(z)

# def naive_add_matrix_and_vector(x, y):
#     assert len(x.shape) == 2
#     assert len(y.shape) == 1
#     assert x.shape[1] == y.shape[0]
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] += y[j]
#     return x

# import numpy as np
# x = np.random.random((64, 3, 32, 10))
# y = np.random.random((32, 10))
# z = np.maximum(x, y)
# print(z)

# import numpy as np
# x = 1235.43
# y = 4656.67
# z = np.dot(x, y)
# print(z)

# def naive_vector_dot(x, y):
#     assert len(x.shape) == 1
#     assert len(y.shape) == 1
#     assert x.shape[0] == y.shape[0]
#     z = 0.
#     for i in range(x.shape[0]):
#         z += x[i] * y[i]
#     print(z)

# import numpy as np
# def naive_matrix_vector_dot(x, y):
#     assert len(x.shape) == 2
#     assert len(y.shape) == 1
#     assert x.shape[1] == y.shape[0]
#     z = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             z[i] += x[i, j] * y[j]
#     return z

# def naive_matrix_vector_dot(x, y):
#     z = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         z[i] = naive_vector_dot(x[i, :], y)
#     return z

# def naive_matrix_dot(x, y):
#     assert len(x.shape) == 2
#     assert len(y.shape) == 2
#     assert x.shape[1] == y.shape[0]
#     z = np.zeros((x.shape[0], y.shape[1]))
#     for i in range(x.shape[0]):
#         for j in range(y.shape[1]):
#             row_x = x[i, :]
#     column_y = y[:, j]
#     z[i, j] = naive_vector_dot(row_x, column_y)
#     return z

# def naive_matrix_vector_dot(x, y):
#     z = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         z[i] = naive_vector_dot(x[i, :], y)
#     return z

# import numpy as np
# x = np.array([[0., 1.],
# [2., 3.],
# [4., 5.]])
# print(x.shape)

# import numpy as np
# x = np.zeros((300, 20))
# x = np.transpose(x)
# print(x.shape)

# past_velocity = 0.
# loss = 0.2
# momentum = 0.1
# while loss > 0.01:
#     w, loss, gradient = get_current_parameters()
# velocity = past_velocity * momentum + learning_rate * gradient
# w = w + momentum * velocity - learning_rate * gradient
# past_velocity = velocity
# update_parameter(w)

# from keras.datasets import imdb
# from keras import models
# from keras import layers
# from keras import optimizers
# from keras import losses
# from keras import metrics
# import matplotlib.pyplot as plt
# import numpy as np
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# def vectorize_sequences(sequences, dimension=10000):
#
#
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results
#
#
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')
#
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='rmsprop',
#             loss='binary_crossentropy',
#             metrics=['accuracy'])
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#             loss='binary_crossentropy',
#             metrics=['accuracy'])
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#                 loss=losses.binary_crossentropy,
#                 metrics=[metrics.binary_accuracy])
#
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
#
# model.compile(optimizer='rmsprop',
#                 loss='binary_crossentropy',
#                 metrics=['acc'])
# history = model.fit(partial_x_train,
#             partial_y_train,
#             epochs=20,
#             batch_size=512,
#             validation_data=(x_val, y_val))
# history_dict = history.history
# plt.clf()
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']
# epochs = range(1, acc + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# # loss_values = history_dict['loss']
# # val_loss_values = history_dict['val_loss']
# # acc = 97
# # epochs = range(1, acc + 1)
# # plt.plot(epochs, loss_values, 'bo', label='Training loss')
# # plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
# history_dict = history.history
# history_dict.keys()
# plt.show()


# reusit

# from keras.datasets import reuters
# import numpy as np
# from keras import models
# from keras import layers
# import matplotlib.pyplot as plt
#
# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results
#
#
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)
#
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))
#
# model.compile(optimizer='rmsprop',
# loss = 'categorical_crossentropy',
# metrics = ['accuracy'])
#
# x_val = x_train[:1000]
# partial_x_train = x_train[1000:]
# y_val = one_hot_train_labels[:1000]
# partial_y_train = one_hot_train_labels[1000:]
#
# history = model.fit(partial_x_train,
# partial_y_train,
# epochs=20,
# batch_size=512,
# validation_data=(x_val, y_val))

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# from keras.datasets import boston_housing
# from keras import models
# from keras import layers
# (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std
# test_data -= mean
# test_data /= std
# def build_model():
#     model = models.Sequential()
#     model.add(layers.Dense(64, activation='relu',
#     input_shape=(train_data.shape[1],)))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(1))
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     return model

# import numpy as np
# import matplotlib.pyplot as plt
# k = 4
# train_data = []
# train_targets = []
# num_val_samples = len(train_data) // k
# num_epochs = 100
# all_scores = []
#
# for i in range(k):
#     print('processing fold #', i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#     partial_train_data = np.concatenate(
#     [train_data[:i * num_val_samples],
#     train_data[(i + 1) * num_val_samples:]],
#     axis=0)
#     partial_train_targets = np.concatenate(
#     [train_targets[:i * num_val_samples],
#     train_targets[(i + 1) * num_val_samples:]],
#     axis=0)
#     model =model.Sequencial()
#     model.fit(partial_train_data, partial_train_targets,
#     epochs=num_epochs, batch_size=1, verbose=0)
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

# from keras import layers
# from keras import models
# from keras.datasets import mnist
# from keras.utils import to_categorical
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = train_images.reshape((60000, 28, 28, 1))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype('float32') / 255
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# model = models.Sequential()
# model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test_acc:', test_acc)

# from keras import layers
# from keras import models
# model_no_max_pool = models.Sequential()
# model_no_max_pool.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model_no_max_pool.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model_no_max_pool.summary()

# import os, shutil
# original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'
# base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
# os.mkdir(base_dir)
# train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
# validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
# train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
# test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
# dst = os.path.join(train_cats_dir, fname)
# shutil.copyfile(src, dst)
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
# dst = os.path.join(validation_cats_dir, fname)
# shutil.copyfile(src, dst)
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
# dst = os.path.join(test_cats_dir, fname)
# shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
# dst = os.path.join(train_dogs_dir, fname)
# shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
# dst = os.path.join(validation_dogs_dir, fname)
# shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
# dst = os.path.join(test_dogs_dir, fname)
# shutil.copyfile(src, dst)
# print('total training cat images:', len(os.listdir(train_cats_dir)))

# from keras import layers
# from keras import models
# from keras import optimizers
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu',
# input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#     optimizer=optimizers.RMSprop(lr=1e-4),
#     metrics=['acc'])
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(
# validation_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50)
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# model.save('cats_and_dogs_small_1.h5')
# model.summary()
# plt.show()
#
# from keras.applications import VGG16
# conv_base = VGG16(weights='imagenet',
# include_top=False,
# input_shape=(150, 150, 3))
# conv_base.summary()

# import os
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
# base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
# test_dir = os.path.join(base_dir, 'test')
# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 20
# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#     directory,
#     target_size=(150, 150),
#     batch_size=batch_size,
#     class_mode='binary')
#     i = 0
# for inputs_batch, labels_batch in generator:
#     features_batch = conv_base.predict(inputs_batch)
# features[i * batch_size : (i + 1) * batch_size] = features_batch
# labels[i * batch_size : (i + 1) * batch_size] = labels_batch
# i += 1
#     if i * batch_size >= sample_count:
# break
# return features, labels

# from keras import models
# from keras import layers
# from keras import optimizers
# import numpy as np
# import matplotlib.pyplot as plt
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
# loss='binary_crossentropy',
# metrics=['acc'])
# history = model.fit(train_features, train_labels,
# epochs=30,
# batch_size=20,
# validation_data=(validation_features, validation_labels))
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# from keras.preprocessing.image import ImageDataGenerator
# from keras import optimizers
# import matplotlib.pyplot as plt
# train_datagen = ImageDataGenerator(
# rescale=1./255,
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest')
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
# train_dir,
# target_size=(150, 150),
# batch_size=20,
# class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(
# validation_dir,
# target_size=(150, 150),
# batch_size=20,
# class_mode='binary')
# model.compile(loss='binary_crossentropy',
# optimizer=optimizers.RMSprop(lr=2e-5),
# metrics=['acc'])
# history = model.fit_generator(
# train_generator,
# steps_per_epoch=100,
# epochs=30,
# validation_data=validation_generator,
# validation_steps=50)
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# img_path = 'C:\Users\Dell\Downloads\download (1).jpg'
# from keras.preprocessing import image
# import numpy as np
# import matplotlib.pyplot as plt
# img = image.load_img(img_path, target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.
# plt.imshow(img_tensor[0])
#
# plt.show()

# from keras.applications import VGG16
# from keras import backend as K
# import numpy as np
# loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
# input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
# step = 1.
# for i in range(40):
#     loss_value, grads_value = iterate([input_img_data])
# input_img_data += grads_value * step
# model = VGG16(weights='imagenet',
# include_top=False)
# layer_name = 'block3_conv1'
# filter_index = 0
# layer_output = model.get_layer(layer_name).output
# loss = K.mean(layer_output[:, :, :, filter_index])
# grads = K.gradients(loss, model.input)[0]
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# iterate = K.function([model.input], [loss, grads])

# def deprocess_image(x):
#     x -= x.mean()
#     x /= (x.std() + 1e-5)
#     x *= 0.1
#     x += 0.5
#     x = np.clip(x, 0, 1)
#     x *= 255
#     x = np.clip(x, 0, 255).astype('uint8')
#     print(x)

# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input, decode_predictions
# model = VGG16(weights='imagenet')
# img_path= 'Users/Dell/Desktop/Untitled111111111.png'
# x = float(img_path)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds, top=3)[0])

# from keras.preprocessing.text import Tokenizer
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# tokenizer = Tokenizer(num_words=1000)
# tokenizer.fit_on_texts(samples)
# sequences = tokenizer.texts_to_sequences(samples)
# one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

# import string
# import numpy as np
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# characters = string.printable
# token_index = dict(zip(range(1, len(characters) + 1), characters))
# max_length = 50
# results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
# for i, sample in enumerate(samples):
#     for j, character in enumerate(sample):
#         index = token_index.get(character)
#         results[i, j, index] = 1.
#         print(results)

# from keras.layers import Embedding
# from keras.datasets import imdb
# import os
# from keras.models import Sequential
# from keras.layers import Flatten, Dens
# from keras import preprocessing
# embedding_layer = Embedding(1000, 64)
# max_features = 10000
# maxlen = 20
# (x_train, y_train), (x_test, y_test) = imdb.load_data(
# num_words=max_features)
# x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# model = Sequential()
# model.add(Embedding(10000, 8, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()
# history = model.fit(x_train, y_train,
# epochs=10,
# batch_size=32,
# validation_split=0.2)

# import numpy as np
# timesteps = 100
# input_features = 32
# output_features = 64
# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features,))
# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features,))
# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t
#     final_output_sequence = np.concatenate(successive_outputs, axis=0)
#     print(final_output_sequence)

# from keras.models import Sequential
# from keras.layers import Embedding, SimpleRNN
# model = Sequential()
# model.add(Embedding(10000, 32))
# model.add(SimpleRNN(32))
# model.summary()
#
# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras.layers import Dense
# import matplotlib.pyplot as plt
# max_features = 10000
# maxlen = 500
# batch_size = 32
# model.add(Embedding(max_features, 32))
# model.add(SimpleRNN(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(input_train, y_train,
# epochs=10,
# batch_size=128,
# validation_split=0.2)
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# print('Loading data...')
# (input_train, y_train), (input_test, y_test) = imdb.load_data(
# num_words=max_features)
# print(len(input_train), 'train sequences')
# print(len(input_test), 'test sequences')
# print('Pad sequences (samples x time)')
# input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
# input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
# print('input_train shape:', input_train.shape)
# print('input_test shape:', input_test.shape)

# from keras.layers import LSTM
# model: site = Sequential()
# model.add(Embedding(max_features, 32))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
# loss='binary_crossentropy',
# metrics=['acc'])
# history = model.fit(input_train, y_train,
# epochs=10,
# batch_size=128,
# validation_split=0.2)

# def generator(data, lookback, delay, min_index, max_index,
# shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#         i = min_index + lookback
# while 1:
#     if shuffle:
#      rows = np.random.randint(
#     min_index + lookback, max_index, size=batch_size)
#     else:
#         if i + batch_size >= max_index:
#
#     i = min_index + lookback
#     rows = np.arange(i, min(i + batch_size, max_index))
#     i += len(rows)
#     samples = np.zeros((len(rows),
#     lookback // step,
#     data.shape[-1]))
#     targets = np.zeros((len(rows),))
#     for j, row in enumerate(rows):
#     indices = range(rows[j] - lookback, rows[j], step)
#     samples[j] = data[indices]
#     targets[j] = data[rows[j] + delay][1]
#     yield samples, targets

# from keras.models import Sequential
# from keras import layers
# import matplotlib.pyplot as plt
# from keras.optimizers import RMSprop
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                                 steps_per_epoch=500,
#                                 epochs=20,
#                                 validation_data=val_gen,
#                                 validation_steps = val_steps)
# model.add(layers.Dense(1))
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()


# from keras.datasets import imdb
# from keras.preprocessing import sequence
#
# max_features = 10000
# max_len = 500
# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=max_len)
# x_test = sequence.pad_sequences(x_test, maxlen=max_len)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)


# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
# model = Sequential()
# model.add(layers.Embedding(max_features, 128, input_length=max_len))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.MaxPooling1D(5))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# model.summary()
# model.compile(optimizer=RMSprop(lr=1e-4),
# loss='binary_crossentropy',
# metrics=['acc'])
# history = model.fit(x_train, y_train,
# epochs=10,
# batch_size=128,
# validation_split=0.2)

# from keras.models import Sequential, Model
# from keras import layers
# from keras import Input
# seq_model = Sequential()
# seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
# seq_model.add(layers.Dense(32, activation='relu'))
# seq_model.add(layers.Dense(10, activation='softmax'))
# input_tensor = Input(shape=(64,))
# x = layers.Dense(32, activation='relu')(input_tensor)
# x = layers.Dense(32, activation='relu')(x)
# output_tensor = layers.Dense(10, activation='softmax')(x)
# model = Model(input_tensor, output_tensor)
# model.summary()

# from keras.models import Model
# from keras import layers
# import numpy as np
# from keras import Input
# text_vocabulary_size = 10000
# question_vocabulary_size = 10000
# answer_vocabulary_size = 500
# text_input = Input(shape=(None,), dtype='int32', name='text')
# embedded_text = layers.Embedding(
# 64, text_vocabulary_size)(text_input)
# encoded_text = layers.LSTM(32)(embedded_text)
# question_input = Input(shape=(None,),
# dtype='int32',
# name='question')
# embedded_question = layers.Embedding(
# 32, question_vocabulary_size)(question_input)
# encoded_question = layers.LSTM(16)(embedded_question)
# concatenated = layers.concatenate([encoded_text, encoded_question],
# axis=-1)
# answer = layers.Dense(answer_vocabulary_size,
# activation='softmax')(concatenated)
# model = Model([text_input, question_input], answer)
# model.compile(optimizer='rmsprop',
# loss='categorical_crossentropy',
# metrics=['acc'])
# num_samples = 1000
# max_length = 100
# text = np.random.randint(1, text_vocabulary_size,
# size=(num_samples, max_length))
# question = np.random.randint(1, question_vocabulary_size,
# size=(num_samples, max_length))
# answers = np.random.randint(0, 1,
# size=(num_samples, answer_vocabulary_size))
# model.fit([text, question], answers, epochs=10, batch_size=128)
# model.fit({'text': text, 'question': question}, answers,
# epochs=10, batch_size=128)

# import keras
# import numpy as np
# class ActivationLogger(keras.callbacks.Callback):
#     def set_model(self, model):
#         self.model = model
#         layer_outputs = [layer.output for layer in model.layers]
#         self.activations_model = keras.models.Model(model.input,
#         layer_outputs)
#     def on_epoch_end(self, epoch, logs=None):
#         if self.validation_data is None:
#             raise RuntimeError('Requires validation_data.')
#
#
# validation_sample = self.validation_data[0][0:1]
# activations = self.activations_model.predict(validation_sample)
# f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
# np.savez(f, activations)
# f.close()

# import numpy as np
# def reweight_distribution(original_distribution, temperature=0.5):
#     distribution = np.log(original_distribution) / temperature
#     distribution = np.exp(distribution)
#     print(distribution / np.sum(distribution))

#

# import random
# import sys
# for epoch in range(1, 60):
#     print('epoch', epoch)
# start_index = random.randint(0, len(text) - maxlen - 1)
# generated_text = text[start_index: start_index + maxlen]
# print('--- Generating with seed: "' + generated_text + '"')
# for temperature in [0.2, 0.5, 1.0, 1.2]:
#     print('------ temperature:', temperature)
# sys.stdout.write(generated_text)
# for i in range(400):
#     sampled = np.zeros((1, maxlen, len(chars)))
# for t, char in enumerate(generated_text):
#     preds = model.predict(sampled, verbose=0)[0]
# next_index = sample(preds, temperature)
# next_char = chars[next_index]
# generated_text += next_char
# generated_text = generated_text[1:]
# sys.stdout.write(next_char)

# import numpy as np
# step = 0.01
# num_octave = 3
# octave_scale = 1.4
# iterations = 20
# max_loss = 10.
# base_image_path = 'C/Users/Dell/Downloads'
# img = preprocess_image(base_image_path)
# original_shape = img.shape[1:3]
# successive_shapes = [original_shape]
# for i in range(1, num_octave):
#     shape = tuple([int(dim / (octave_scale ** i))
# for dim in original_shape])
#     successive_shapes.append(shape)
#     successive_shapes = successive_shapes[::-1]
#     original_img = np.copy(img)
#     shrunk_original_img = resize_img(img, successive_shapes[0])
# for shape in successive_shapes:
#     print('Processing image shape', shape)
# img = resize_img(img, shape)
# img = gradient_ascent(img,
# iterations=iterations,
# step=step,
# max_loss=max_loss)
# upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
# same_size_original = resize_img(original_img, shape)
# lost_detail = same_size_original - upscaled_shrunk_original_img
# img += lost_detail
# shrunk_original_img = resize_img(original_img, shape)
# save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
# save_img(img, fname='final_dream.png')

# import scipy
# from keras.preprocessing import image
#
#
# def resize_img(img, size):
#     img = np.copy(img)
#
#
# factors = (1,
#            float(size[0]) / img.shape[1],
#            float(size[1]) / img.shape[2],
#            1)
#     return scipy.ndimage.zoom(img, factors, order=1)
#
#
# def save_img(img, fname):
#     pil_img = deprocess_image(np.copy(img))
#     scipy.misc.imsave(fname, pil_img)
#
#
# def preprocess_image(image_path):
#     img = image.load_img(image_path)
#     img = image.img_to_array(img)
#
# img = np.expand_dims(img, axis=0)
# img = inception_v3.preprocess_input(img)
# return img
# def deprocess_image(x):
#     if K.image_data_format() == 'channels_first':
#         x = x.reshape((3, x.shape[2], x.shape[3]))
#         x = x.transpose((1, 2, 0))
#     else:
#         x = x.reshape((x.shape[1], x.shape[2], 3))
#         x /= 2.
#         x += 0.5
#         x *= 255.
#         x = np.clip(x, 0, 255).astype('uint8')
# return x

#

# import os
# from keras.preprocessing import image
# (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
# x_train = x_train[y_train.flatten() == 6]
# x_train = x_train.reshape(
# (x_train.shape[0],) +
# (height, width, channels)).astype('float32') / 255.
# iterations = 10000
# batch_size = 20
# save_dir = 'your_dir'
# start = 0
# for step in range(iterations):
#     random_latent_vectors = np.random.normal(size=(batch_size,
# latent_dim))
# generated_images = generator.predict(random_latent_vectors)
# stop = start + batch_size
# real_images = x_train[start: stop]
# combined_images = np.concatenate([generated_images, real_images])
# labels = np.concatenate([np.ones((batch_size, 1)),
# np.zeros((batch_size, 1))])
# labels += 0.05 * np.random.random(labels.shape)
# d_loss = discriminator.train_on_batch(combined_images, labels)
# random_latent_vectors = np.random.normal(size=(batch_size,
# latent_dim))
# misleading_targets = np.zeros((batch_size, 1))
# a_loss = gan.train_on_batch(random_latent_vectors,
# misleading_targets)
# start += batch_size
# if start > len(x_train) - batch_size:
#     start = 0
# if step % 100 == 0:
#     gan.save_weights('gan.h5')
# print('discriminator loss:', d_loss)
# print('adversarial loss:', a_loss)
# img = image.array_to_img(generated_images[0] * 255., scale=False)
# img.save(os.path.join(save_dir,
# 'generated_frog' + str(step) + '.png'))
# img = image.array_to_img(real_images[0] * 255., scale=False)
# img.save(os.path.join(save_dir,
# 'real_frog' + str(step) + '.png'))

