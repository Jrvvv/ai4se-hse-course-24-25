from pathlib import Path
import datasets
import pandas as pd
from .dataset.data import CodeReviewDataPreprocessor


def prepare(raw_data: Path, output_excel_path: Path) -> datasets.Dataset:
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
    preprocessor.preprocess_dataset()

    # СОХРАНЕНИЕ В EXCEL
    output_excel = output_excel_path
    preprocessor.df.to_excel(output_excel, index=False)
    print(f"Очищенные данные сохранены в '{output_excel}'")
