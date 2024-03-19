import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras import regularizers
from keras.callbacks import EarlyStopping
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Подключение к базе данных PostgreSQL
engine = create_engine('postgresql://postgres:8196@localhost:5432/diploma1')

# Запрос для извлечения данных из таблицы
query = "SELECT domain, class FROM my_dataset_all_2"

# Загрузка датасета из таблицы в DataFrame
df = pd.read_sql(query, engine)

# Преобразование меток в числовой формат
df['class'] = df['class'].astype('category').cat.codes

# Разделение на обучающую, валидационную и тестовую выборки
train_data, test_data, train_labels, test_labels = train_test_split(df['domain'], df['class'],
                                                                    test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                  test_size=0.1, random_state=42)

# Создание словаря символов
chars = sorted(list(set(''.join(train_data))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Преобразование символов в индексы для обучающей, валидационной и тестовой выборок
train_data_idx = [[char_indices[char] for char in domain] for domain in train_data]
val_data_idx = [[char_indices[char] for char in domain] for domain in val_data]
test_data_idx = [[char_indices[char] for char in domain] for domain in test_data]

# Создание последовательности фиксированной длины
maxlen = max(len(domain) for domain in train_data)
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_data_idx, maxlen=maxlen)
val_X = tf.keras.preprocessing.sequence.pad_sequences(val_data_idx, maxlen=maxlen)
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_data_idx, maxlen=maxlen)

# Преобразование меток в one-hot encoding
num_classes = len(set(train_labels))
train_labels = to_categorical(train_labels, num_classes=num_classes)
val_labels = to_categorical(val_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 128, input_length=maxlen),
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.GRU(128, kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ранняя остановка
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Обучение модели на обучающей выборке с использованием регуляризации и ранней остановки
history = model.fit(train_X, train_labels, batch_size=128, epochs=10, verbose=1,
                    validation_data=(val_X, val_labels), callbacks=[early_stop])

# Сохранение модели
model.save('dga_detection_model_3_regularized_fin_9.h5')

# Оценка качества модели на тестовой выборке
loss, accuracy = model.evaluate(test_X, test_labels, verbose=1)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')


# График обучения модели
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# График потерь модели
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Закрытие соединения с базой данных
engine.dispose()
