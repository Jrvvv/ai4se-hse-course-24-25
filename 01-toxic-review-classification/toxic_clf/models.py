from statistics import mean, stdev
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Импортируем напрямую из файла models.py в подпапке models
from toxic_clf.models.models import CodeReviewClassifier


def classifier(dataset, model_name):
    """
    Классификатор с совместимостью с интерфейсом из main.py
    """
    print(f"Запуск классификации с моделью: {model_name}")
    print(f"Размер датасета: {len(dataset)}")
    
    # Сохраняем датасет во временный Excel файл для CodeReviewClassifier
    temp_excel = Path("temp_cleaned_reviews.xlsx")
    
    # Создаем DataFrame из datasets.Dataset
    df_data = {
        'cleaned_text': dataset['text'],
        'original_text': dataset['original_text']
    }
    
    # Добавляем метки если они есть
    if 'labels' in dataset.column_names:
        df_data['is_toxic'] = dataset['labels']
    else:
        # Если меток нет, создаем фиктивные для совместимости
        print("Внимание: метки не найдены в датасете. Созданы фиктивные метки.")
        df_data['is_toxic'] = [0] * len(dataset)
    
    df = pd.DataFrame(df_data)
    df.to_excel(temp_excel, index=False)
    print(f"Временный файл создан: {temp_excel}")
    
    # Инициализируем классификатор
    clf = CodeReviewClassifier(str(temp_excel))
    
    # Запускаем соответствующий пайплайн в зависимости от модели
    if model_name == 'classic_ml':
        print("=== ЗАПУСК КЛАССИЧЕСКИХ МОДЕЛЕЙ ===")
        clf.prepare_data()
        clf.vectorize_text(method='both')
        clf.train_classical_models()
        clf.evaluate_classical_models()
        clf.hyperparameter_tuning()
        
    elif model_name == 'roberta':
        print("=== ЗАПУСК RoBERTa ===")
        clf.prepare_data()
        clf.train_roberta()
        
    elif model_name == 'microsoft/codebert-base':
        print("=== ЗАПУСК CodeBERT ===")
        clf.prepare_data()
        clf.train_codebert()
    
    # Генерируем отчет
    report = clf.generate_report()
    
    # Очищаем временный файл
    if temp_excel.exists():
        temp_excel.unlink()
        print(f"Временный файл удален: {temp_excel}")
    
    return report