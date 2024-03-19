from django.db import models
import tensorflow as tf
import numpy as np

# Определение уникальных меток классов
unique_labels = ['dga', 'legit']

# Создание словаря символов
chars = sorted(list(set('abcdefghijklmnopqrstuvwxyz0123456789-._')))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

class OneDomainPredictor(models.Model):
    class Meta:
        verbose_name = 'Domain Predictor'
        verbose_name_plural = 'Domain Predictors'

    # Загрузка модели из файла
    model = tf.keras.models.load_model('dga_detection_model_3_regularized.h5')

    domain_name = models.CharField(max_length=255)
    predicted_class = models.CharField(max_length=255)
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.domain_name

    # Функция для преобразования домена в последовательность индексов символов
    def domain_to_indices(self, domain):
        return [char_indices[char] for char in domain]

    # Функция для преобразования последовательности индексов символов в домен
    def indices_to_domain(self, indices):
        return ''.join(indices_char[idx] for idx in indices)

    # Метод для предсказания класса по домену
    def predict_one(self, domain):
        # Преобразование домена в последовательность индексов символов
        domain_indices = self.domain_to_indices(domain)
        # Создание последовательности фиксированной длины
        maxlen = self.model.input_shape[1]
        X = tf.keras.preprocessing.sequence.pad_sequences([domain_indices], maxlen=maxlen)
        # Предсказание класса
        y_pred = self.model.predict(X)
        class_idx = np.argmax(y_pred)
        class_label = unique_labels[class_idx]
        return class_label
