# ML Course Project: Wine Quality

Репозиторий содержит ML-проект по исследованию и прогнозированию качества вина на основе его физико-химических свойств. Проект развивается в рамках сквозного курса (ДЗ 1–6).

## Описание датасета и лицензия

- **Датасет:** Wine Quality Dataset (Cortez et al., 2009)
- **Источник:** [UCI Machine Learning Repository: Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Лицензия:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)
- **Описание задачи:** Регрессия для оценки качества вина (`quality`, шкала от 0 до 10) по 11 физико-химическим характеристикам (кислотность, остаточный сахар, хлориды, плотность, pH, сульфаты, алкоголь и др.).
- **Размер датасета:** 1 599 объектов (red wine), 12 колонок.

## Структура проекта

```text
ml-hw/
├── .github/                  # Настройки CI/CD (при необходимости)
├── .gitignore                # Игнорируемые файлы (данные, артефакты, кэш, .pdf)
├── .pre-commit-config.yaml   # Настройки pre-commit хуков (Ruff, YAML, whitespace)
├── pyproject.toml            # Описание проекта, зависимости и конфиг Ruff
├── uv.lock                   # Точно зафиксированные версии зависимостей
├── README.md                 # Документация и инструкция по запуску
├── data/
│   └── raw/                  # Каталог для сырых данных (не коммитится в Git)
├── artifacts/                # Сохраненные модели (model.pkl) и метрики (metrics.json)
├── reports/
│   ├── hw-1.md               # Отчет по выполнению ДЗ 1
│   └── screenshots/          # Скриншоты работы с инструментами
└── src/
    ├── __init__.py
    ├── get_data.py           # Скрипт загрузки и валидации датасета
    └── train.py              # Скрипт обучения модели и расчета метрик
```

## Требования к окружению

- **Python:** `>= 3.11`
- **Пакетный менеджер:** [uv](https://docs.astral.sh/uv/) `>= 0.13`
- **Git**

## Инструкция по установке и запуску

### 1. Клонирование репозитория

```bash
git clone https://github.com/YvajeniePj/ml-hw.git
cd ml-hw
```

### 2. Установка зависимостей

Проект использует менеджер пакетов `uv` и версионирование через lock-файл:

```bash
# Установка основного окружения и dev-зависимостей из uv.lock
uv sync --all-extras
```

### 3. Проверка качества кода (Ruff и pre-commit)

В репозитории настроен линтер Ruff и pre-commit хуки:

```bash
# Установка pre-commit хуков в Git
uv run pre-commit install

# Ручной запуск всех проверок (Ruff lint, format, yaml, trailing-whitespace)
uv run pre-commit run --all-files
```

Для проверки только с помощью Ruff:

```bash
uv run ruff check .
uv run ruff format --check .
```

### 4. Загрузка данных

Скрипт загружает датасет из репозитория UCI, сохраняет его в `data/raw/winequality-red.csv` и проводит валидацию схемы:

```bash
uv run python -m src.get_data
```

*Примечание:* Если источнику требуется токен доступа, скрипт автоматически считывает его из переменной окружения `DATASET_DOWNLOAD_TOKEN` (токен не коммитится в репозиторий).

### 5. Обучение модели

Скрипт считывает сырые данные, обучает модель `RandomForestRegressor`, оценивает качество на тестовой выборке (RMSE, MAE, R²) и сохраняет результаты:

```bash
uv run python -m src.train
```

После выполнения команды формируются артефакты:
- `artifacts/model.pkl` — обученная модель
- `artifacts/metrics.json` — сохраненные метрики в формате JSON

Пример содержимого `artifacts/metrics.json`:
```json
{
  "mse": 0.3297,
  "rmse": 0.5742,
  "mae": 0.4182,
  "r2": 0.4954
}
```
