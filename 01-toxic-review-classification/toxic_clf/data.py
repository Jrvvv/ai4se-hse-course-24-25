from pathlib import Path
import datasets
import pandas as pd
from .dataset.data import CodeReviewDataPreprocessor


def prepare(raw_data: Path, output_excel_path: Path) -> datasets.Dataset:
    """
    Dataset preparation using CodeReviewDataPreprocessor
    """
    # Checking file extension
    if raw_data.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError(f"Unsupported file format: {raw_data.suffix}. Excel file (.xlsx, .xls) expected")

    # Preprocessor initialization
    preprocessor = CodeReviewDataPreprocessor()

    # Loading data from Excel
    preprocessor.load_excel_data(
        filepath=str(raw_data),
        text_column='review_text',
        label_column='label'
    )

    # Analyzing obscene words distribution
    preprocessor.explore_obscene_words_distribution()

    # Cleaning and preprocessing
    preprocessor.clean_data()
    preprocessor.preprocess_dataset()

    # SAVING TO EXCEL
    output_excel = output_excel_path
    preprocessor.df.to_excel(output_excel, index=False)
