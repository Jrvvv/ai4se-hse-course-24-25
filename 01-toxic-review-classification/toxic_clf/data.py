from pathlib import Path
import datasets
import pandas as pd
from .dataset.data import CodeReviewDataPreprocessor


def prepare(raw_data: Path) -> datasets.Dataset:
    """
    Подготовка датасета с использованием CodeReviewDataPreprocessor
    """
    # Проверяем расширение файла
    if raw_data.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError(f"Неподдерживаемый формат файла: {raw_data.suffix}. Ожидается Excel файл (.xlsx, .xls)")
    
    # Инициализация препроцессора
    preprocessor = CodeReviewDataPreprocessor()

    # Загрузка данных из Excel
    preprocessor.load_excel_data(
        filepath=str(raw_data), 
        text_column='review_text', 
        label_column='label'
    )
    
    # Анализ распределения obscene слов
    preprocessor.explore_obscene_words_distribution()

    # Очистка и предобработка
    preprocessor.clean_data()
    preprocessor.preprocess_dataset(show_examples=True)

    # СОХРАНЕНИЕ В EXCEL
    output_excel = Path("cleaned_code_reviews.xlsx")
    preprocessor.df.to_excel(output_excel, index=False)
    print(f"Очищенные данные сохранены в '{output_excel}'")

    # Cоздаем datasets.Dataset из обработанных данных для дальнейшего использования
    dataset_dict = {
        'text': preprocessor.df['cleaned_text'].tolist(),
        'original_text': preprocessor.df['original_text'].tolist(),
    }
    
    # Добавляем labels если они есть
    if preprocessor.label_column and preprocessor.label_column in preprocessor.df.columns:
        dataset_dict['labels'] = preprocessor.df[preprocessor.label_column].tolist()
        print(f"Метки обнаружены. Распределение: {preprocessor.df[preprocessor.label_column].value_counts().to_dict()}")
    
    # Создаем datasets.Dataset
    dataset = datasets.Dataset.from_dict(dataset_dict)
    
    print(f"Подготовленный датасет для ML: {len(dataset)} примеров")
    print(f"Колонки: {dataset.column_names}")
    
    return dataset


def load_dataset(path: Path) -> datasets.Dataset:
    """
    Загрузка подготовленного датасета с диска
    """
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден по пути: {path}")
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    """
    Сохранение датасета на диск
    """
    # Создаем директорию если не существует
    path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    print(f"Датасет для ML сохранен в: {path}")