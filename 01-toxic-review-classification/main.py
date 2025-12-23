import argparse
from pathlib import Path
from toxic_clf.data import prepare
from toxic_clf.models import classifier

def main():
    args = parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        print("Unknown command. Use --help for help.")

def parse_args():
    parser = argparse.ArgumentParser(description='Toxic comment classification in code review')
    subparsers = parser.add_subparsers(dest='cmd', required=True, help='Available commands')

    default_clean_data_path = Path('./prepared-dataset/cleaned_code_reviews.xlsx')
    # Parser for data preparation
    prepare_data_parser = subparsers.add_parser('prepare-data', help='Dataset preparation')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        'input',
        help='Path to source dataset (Excel file)',
        type=Path,
    )
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset',
        type=Path,
        default=default_clean_data_path,
    )

    # Parser for classification
    predict_parser = subparsers.add_parser('classify', help='Comment classification')
    predict_parser.set_defaults(func=classify)
    predict_parser.add_argument(
        '-d',
        '--dataset_path',
        help='Path to prepared dataset',
        type=Path,
        default=default_clean_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        choices=['classic_ml', 'codebert-base', 'roberta'],
        default='classic_ml',
        help='Model for classification'
    )

    return parser.parse_args()

def prepare_data(args):
    """Dataset preparation with error handling"""
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"File {args.input} not found")

        print(f"Loading data from: {args.input}")
        print(f"Saving data to: {args.output}")

        args.output.parent.mkdir(parents=True, exist_ok=True)

        prepare(args.input, args.output)

        print("Data preparation completed successfully!")

    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise

def classify(args):
    """Classification with error handling"""
    try:
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"Prepared dataset {args.dataset_path} not found. Run prepare-data first.")

        print(f"Running classification with model: {args.model}")
        result = classifier(args.dataset_path, args.model)

        print("Classification completed successfully!")
        return result

    except Exception as e:
        print(f"Error during classification: {e}")
        raise

if __name__ == '__main__':
    main()
