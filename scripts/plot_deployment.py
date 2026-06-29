"""plot_deployment.py — схема развертывания (готовая картинка, серо-синяя гамма).

Без служебных обозначений: только блоки проекта и связи, включая источник данных
(архивные коллекции + загрузка через браузер).

Использование:
    python scripts/plot_deployment.py --out docs/figures/fig_deployment.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Весь текст — Times New Roman (с serif-фолбэком, если шрифта нет).
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

EDGE = "#34495e"
C_SERVER = "#eef2f6"
C_APP = "#cfdde9"
C_STORE = "#e3eaf1"
C_EXT = "#dbe4ec"
C_USER = "#d6dee6"


def box(ax, x, y, w, h, text, fc, fs=10, bold=False, radius=0.08):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       fc=fc, ec=EDGE, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#1c2833")


def arrow(ax, p1, p2, text="", fs=8, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                        lw=1.3, color=EDGE, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom", fontsize=fs,
                color="#34495e", style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/fig_deployment.png")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.8); ax.axis("off")

    # клиент
    box(ax, 0.5, 5.4, 2.2, 0.85, "Сотрудник\nбиблиотеки", C_USER, fs=10)
    box(ax, 0.5, 3.9, 2.2, 0.9, "Веб-браузер\n(интерфейс)", C_USER, fs=10)
    # источник данных
    box(ax, 0.5, 1.0, 2.4, 1.0, "Архивные коллекции\n(НЭБ, NYPL)", C_EXT, fs=10)

    # сервер (контейнер)
    box(ax, 3.5, 0.6, 5.9, 5.7, "", C_SERVER, radius=0.06)
    ax.text(6.45, 5.95, "Сервер (контейнер)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1c2833")
    box(ax, 4.0, 4.0, 4.9, 1.5,
        "Веб-приложение\nAPI + пайплайн\n(BLIP → перевод → SigLIP → описание, OCR)",
        C_APP, fs=10)
    box(ax, 3.9, 1.2, 1.7, 1.0, "Изображения", C_STORE, fs=9.5)
    box(ax, 5.75, 1.2, 1.7, 1.0, "descriptions.json", C_STORE, fs=9.5)
    box(ax, 7.6, 1.2, 1.6, 1.0, "Кэш моделей", C_STORE, fs=9.5)

    # внешние модели
    box(ax, 10.0, 3.9, 1.8, 1.6, "Репозитории\nмоделей\n(HuggingFace,\nPaddleX)", C_EXT, fs=9.5)

    # связи
    arrow(ax, (1.6, 5.4), (1.6, 4.85), "")                                  # сотрудник -> браузер
    arrow(ax, (2.7, 4.35), (4.0, 4.75), "изображение / правка")             # браузер -> приложение
    arrow(ax, (2.9, 1.6), (4.3, 1.7), "исходные\nданные")                   # коллекции -> изображения
    arrow(ax, (4.75, 2.2), (5.5, 4.0), "чтение")                            # изображения -> приложение
    arrow(ax, (8.4, 2.2), (7.4, 4.0), "модели")                            # кэш -> приложение
    arrow(ax, (6.6, 4.0), (6.6, 2.2), "сохранение\nописания")              # приложение -> descriptions
    arrow(ax, (10.0, 4.3), (9.2, 2.0), "загрузка\nвесов", rad=-0.15)        # репозитории -> кэш

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("->", out)


if __name__ == "__main__":
    main()
