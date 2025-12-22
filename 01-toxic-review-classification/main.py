import argparse
from pathlib import Path
from toxic_clf.data import prepare
from toxic_clf.models import classifier


def main():
    args = parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        print("Неизвестная команда. Используйте --help для справки.")


def parse_args():
    parser = argparse.ArgumentParser(description='Классификация токсичных комментариев в code review')
    subparsers = parser.add_subparsers(dest='cmd', required=True, help='Доступные команды')

    default_clean_data_path = Path('./prepared-dataset/cleaned_code_reviews.xlsx')
    # Парсер для подготовки данных
    prepare_data_parser = subparsers.add_parser('prepare-data', help='Подготовка датасета')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        'input',
        help='Путь к исходному датасету (Excel файл)',
        type=Path,
    )
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Путь для сохранения подготовленного датасета',
        type=Path,
        default=default_clean_data_path,
    )

    # Парсер для классификации
    predict_parser = subparsers.add_parser('classify', help='Классификация комментариев')
    predict_parser.set_defaults(func=classify)
    predict_parser.add_argument(
        '-d',
        '--dataset_path',
        help='Путь к подготовленному датасету',
        type=Path,
        default=default_clean_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        choices=['classic_ml', 'microsoft/codebert-base', 'roberta'],
        default='classic_ml',
        help='Модель для классификации'
    )

    return parser.parse_args()


def prepare_data(args):
    """Подготовка датасета с обработкой ошибок"""
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"Файл {args.input} не найден")
        
        print(f"Загрузка данных из: {args.input}")
        print(f"Сохранение данных в: {args.output}")

        args.output.parent.mkdir(parents=True, exist_ok=True)

        prepare(args.input, args.output)

        print("Подготовка данных завершена успешно!")

    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        raise


def classify(args):
    """Классификация с обработкой ошибок"""
    try:
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"Подготовленный датасет {args.dataset_path} не найден. Сначала выполните prepare-data.")

        print(f"Запуск классификации с моделью: {args.model}")
        result = classifier(args.dataset_path, args.model)
        
        print("Классификация завершена успешно!")
        return result
        
    except Exception as e:
        print(f"Ошибка при классификации: {e}")
        raise


if __name__ == '__main__':
    main()