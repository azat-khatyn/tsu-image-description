"""Тесты CLIPScorer без загрузки моделей.

cosine — чистая функция; score проверяется через object.__new__ с подменой
encode_image/encode_text (модели не нужны), по образцу остальных тестов.
"""

import numpy as np

from tsu_image_description.clip_scorer import CLIPScorer


def test_cosine_identical_vectors():
    v = np.array([0.6, 0.8])  # уже единичный
    assert abs(CLIPScorer.cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(CLIPScorer.cosine(a, b)) < 1e-9


def test_cosine_is_dot_product():
    a = np.array([0.5, 0.5, 0.5, 0.5])
    b = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(CLIPScorer.cosine(a, b) - 0.5) < 1e-9


def test_score_combines_encoders():
    # модель не грузится: подменяем энкодеры
    s = object.__new__(CLIPScorer)
    s.encode_image = lambda p: np.array([1.0, 0.0])
    s.encode_text = lambda t, lang="ru": np.array([1.0, 0.0]) if lang == "ru" else np.array([0.0, 1.0])

    assert abs(s.score("x.jpg", "вид на пруд", "ru") - 1.0) < 1e-9
    assert abs(s.score("x.jpg", "a pond view", "en")) < 1e-9
