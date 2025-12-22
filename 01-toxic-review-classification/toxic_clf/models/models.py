import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import accelerate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
from tqdm import tqdm
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Logging setup for transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


class CodeReviewClassifier:
    def __init__(self, data_path):
        """Classifier initialization"""
        self.df = self.load_excel_data(data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.vectorizers = {}
        self.models = {}
        self.results = {}

    def load_excel_data(self, data_path):
        """Loading data from Excel file with specific columns is_toxic and cleaned_text"""
        try:
            df = pd.read_excel(data_path)
            print(f"File loaded successfully: {data_path}")
            print(f"Data size: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Checking for required columns
            required_columns = ['is_toxic', 'cleaned_text']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Error: missing required columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                raise ValueError(f"Missing columns: {missing_columns}")

            # Data quality check
            print(f"\nData check:")
            print(f"Number of records: {len(df)}")
            print(f"Missing values in cleaned_text: {df['cleaned_text'].isnull().sum()}")
            print(f"Missing values in is_toxic: {df['is_toxic'].isnull().sum()}")
            print(f"Unique values in is_toxic: {df['is_toxic'].unique()}")
            print(f"Label distribution:\n{df['is_toxic'].value_counts()}")

            # Data cleaning
            df_clean = df.dropna(subset=['cleaned_text', 'is_toxic']).copy()
            df_clean = df_clean[df_clean['cleaned_text'].astype(str).str.strip().str.len() > 0]

            # Convert labels to numeric format if needed
            if df_clean['is_toxic'].dtype == 'object':
                df_clean['is_toxic'] = df_clean['is_toxic'].astype(int)

            print(f"Data after cleaning: {len(df_clean)} records")
            print(f"Final label distribution:\n{df_clean['is_toxic'].value_counts()}")

            return df_clean

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def prepare_data(self, test_size=0.2, random_state=42):
        """Data preparation and splitting"""
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['cleaned_text'],
            self.df['is_toxic'], test_size=test_size, random_state=random_state, stratify=self.df['is_toxic'])

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Class distribution in training set:\n{pd.Series(self.y_train).value_counts()}")

    # CLASSICAL MODELS
    def vectorize_text(self, method='both'):
        """Text vectorization using CountVectorizer and TfidfVectorizer"""
        vectorizers = {}

        if method in ['count', 'both']:
            # CountVectorizer
            count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2,
                max_df=0.8)
            X_train_count = count_vectorizer.fit_transform(self.X_train.astype(str))
            X_test_count = count_vectorizer.transform(self.X_test.astype(str))
            vectorizers['count'] = {'vectorizer': count_vectorizer, 'X_train': X_train_count, 'X_test': X_test_count}
            print("CountVectorizer completed")

        if method in ['tfidf', 'both']:
            # TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2,
                max_df=0.8, sublinear_tf=True)
            X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train.astype(str))
            X_test_tfidf = tfidf_vectorizer.transform(self.X_test.astype(str))
            vectorizers['tfidf'] = {'vectorizer': tfidf_vectorizer, 'X_train': X_train_tfidf, 'X_test': X_test_tfidf}
            print("TfidfVectorizer completed")

        self.vectorizers = vectorizers
        return vectorizers

    def train_classical_models(self):
        """Training classical models with cross-validation"""
        models = {}
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        for vec_name, vec_data in self.vectorizers.items():
            print(f"\n--- Training models with {vec_name.upper()} ---")

            X_train = vec_data['X_train']
            X_test = vec_data['X_test']

            # Logistic Regression
            lr_params = {'C': 0.1, 'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}
            lr = LogisticRegression(**lr_params)

            # Cross-validation
            lr_scores = cross_val_score(lr, X_train, self.y_train, cv=kfold, scoring='f1_weighted')
            print(f"Logistic Regression CV F1-score: {lr_scores.mean():.4f} (+/- {lr_scores.std() * 2:.4f})")

            # Training on full training set
            lr.fit(X_train, self.y_train)
            lr_pred = lr.predict(X_test)

            # Random Forest
            rf_params = {'n_estimators': 100, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1,
                'class_weight': 'balanced'}
            rf = RandomForestClassifier(**rf_params)

            # Cross-validation
            rf_scores = cross_val_score(rf, X_train, self.y_train, cv=kfold, scoring='f1_weighted')
            print(f"Random Forest CV F1-score: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

            # Training on full training set
            rf.fit(X_train, self.y_train)
            rf_pred = rf.predict(X_test)

            models[f'lr_{vec_name}'] = {'model': lr, 'predictions': lr_pred, 'cv_scores': lr_scores}

            models[f'rf_{vec_name}'] = {'model': rf, 'predictions': rf_pred, 'cv_scores': rf_scores}

            # Confusion matrices
            print(f"\nConfusion matrix for Logistic Regression ({vec_name}):")
            self.plot_confusion_matrix(f'LR_{vec_name}', lr_pred)

            print(f"\nConfusion matrix for Random Forest ({vec_name}):")
            self.plot_confusion_matrix(f'RF_{vec_name}', rf_pred)

        self.models.update(models)
        return models

    def evaluate_classical_models(self):
        """Evaluation of classical models"""
        for model_name, model_data in self.models.items():
            if not model_name.startswith(('lr_', 'rf_')):
                continue

            y_pred = model_data['predictions']
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

            self.results[model_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

            print(f"\n--- {model_name.upper()} ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

    def plot_confusion_matrix(self, model_name, y_pred):
        """Plotting confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'],
                    yticklabels=['Non-Toxic', 'Toxic'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Metrics for each class
        print(classification_report(self.y_test, y_pred, target_names=['Non-Toxic', 'Toxic']))

    def hyperparameter_tuning(self):
        """Hyperparameter tuning for classical models"""
        print("\n--- Hyperparameter tuning ---")

        if 'tfidf' in self.vectorizers:
            X_train = self.vectorizers['tfidf']['X_train']

            # Tuning for Logistic Regression
            lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}

            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            lr_grid_search = GridSearchCV(lr, lr_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            lr_grid_search.fit(X_train, self.y_train)

            print(f"Best parameters for Logistic Regression: {lr_grid_search.best_params_}")
            print(f"Best F1-score: {lr_grid_search.best_score_:.4f}")

            # Tuning for Random Forest
            rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]}

            rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            rf_grid_search.fit(X_train, self.y_train)

            print(f"Best parameters for Random Forest: {rf_grid_search.best_params_}")
            print(f"Best F1-score: {rf_grid_search.best_score_:.4f}")

    # TRANSFORMER MODELS
    def create_tokenize_function(self, tokenizer):
        """Creating a serializable tokenization function"""

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding=True, truncation=True, max_length=256, return_tensors="pt")

        return tokenize_function

    def prepare_transformer_data(self, model_name='roberta-base'):
        """Data preparation for transformers with improved error handling"""
        print(f"Loading tokenizer for {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Adding padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Creating a serializable tokenization function
            tokenize_function = self.create_tokenize_function(self.tokenizer)

            # Creating datasets
            train_dataset = Dataset.from_dict(
                {'text': self.X_train.astype(str).tolist(), 'label': self.y_train.tolist()})

            test_dataset = Dataset.from_dict({'text': self.X_test.astype(str).tolist(), 'label': self.y_test.tolist()})

            # Tokenization with caching disabled to avoid warnings
            self.train_tokenized = train_dataset.map(tokenize_function, batched=True, load_from_cache_file=False
                # Disabling caching to avoid warnings
            )
            self.test_tokenized = test_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

            return self.train_tokenized, self.test_tokenized

        except Exception as e:
            print(f"Error preparing data for {model_name}: {e}")
            raise

    def compute_metrics(self, eval_pred):
        """Computing metrics for transformers"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def train_transformer(self, model_name='roberta-base', model_display_name='RoBERTa'):
        """Training transformer model with improved error handling"""
        print(f"\n--- Training {model_display_name} ---")

        try:
            # Data preparation
            self.prepare_transformer_data(model_name)

            # Model definition
            num_labels = len(np.unique(self.y_train))
            print(f"Loading model {model_name} for {num_labels} classes...")

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                ignore_mismatched_sizes=True  # Ignore size mismatches
            )

            # GPU availability check
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            print(f"Using device: {device}")

            # Training parameters optimized to avoid memory overflow
            training_args = TrainingArguments(output_dir=f'./results_{model_display_name}', num_train_epochs=1,
                per_device_train_batch_size=4,  # Reduced batch size
                per_device_eval_batch_size=4, warmup_steps=50,  # Reduced warmup steps
                weight_decay=0.01, logging_dir=f'./logs_{model_display_name}', logging_steps=10,
                # Increased to reduce output
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                metric_for_best_model="f1", save_total_limit=1,  # Reduced to save space
                dataloader_pin_memory=False,  # May help with memory
                gradient_accumulation_steps=2,  # Gradient accumulation
                fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
            )

            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Trainer
            trainer = Trainer(model=model, args=training_args, train_dataset=self.train_tokenized,
                eval_dataset=self.test_tokenized, tokenizer=self.tokenizer, data_collator=data_collator,
                compute_metrics=self.compute_metrics, )

            # Training
            print(f"Starting training for {model_display_name}...")
            trainer.train()

            # Prediction on test data
            print(f"Predicting on test data...")
            predictions = trainer.predict(self.test_tokenized)
            y_pred = np.argmax(predictions.predictions, axis=1)

            # Saving results
            self.models[model_display_name] = {'model': trainer, 'predictions': y_pred}

            # Computing metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

            self.results[model_display_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall,
                'f1_score': f1}

            print(f"\n--- {model_display_name} Results ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            # Confusion matrix
            print(f"\nConfusion matrix for {model_display_name}:")
            self.plot_confusion_matrix(model_display_name, y_pred)

            return trainer

        except Exception as e:
            print(f"Error training {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_roberta(self):
        """Training RoBERTa"""
        return self.train_transformer('roberta-base', 'RoBERTa')

    def train_codebert(self):
        """Training CodeBERT"""
        return self.train_transformer('microsoft/codebert-base', 'CodeBERT')

    def generate_report(self):
        """Generating report with model comparison"""
        print("\n" + "=" * 60)
        print("FINAL CLASSIFICATION MODELS REPORT")
        print("=" * 60)

        # Creating DataFrame with results
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df = results_df.round(4)

        # Sorting by F1-score
        results_df = results_df.sort_values('f1_score', ascending=False)

        print("\nModel comparison (sorted by F1-score):")
        print(results_df)

        # Visualization
        plt.figure(figsize=(12, 8))

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = range(len(results_df))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
            plt.bar(x, results_df[metric], color=colors[:len(results_df)])
            plt.title(f'{metric.capitalize()} Comparison')
            plt.xticks(x, results_df.index, rotation=45)
            plt.ylim(0, 1)
            for j, v in enumerate(results_df[metric]):
                plt.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        best_model = results_df.iloc[0]
        print(f"\n BEST MODEL: {results_df.index[0]}")
        print(f" F1-Score: {best_model['f1_score']:.4f}")
        print(f" Accuracy: {best_model['accuracy']:.4f}")
        print(f" Precision: {best_model['precision']:.4f}")
        print(f" Recall: {best_model['recall']:.4f}")

        return results_df

    def load_from_datasets(self, dataset):
        """Loading data from datasets.Dataset"""
        try:
            # Creating DataFrame from datasets.Dataset
            df_data = {
                'cleaned_text': dataset['text'],
                'original_text': dataset['original_text']
            }

            # Adding labels if they exist
            if 'labels' in dataset.column_names:
                df_data['is_toxic'] = dataset['labels']
            else:
                print("Warning: labels not found. Created dummy labels.")
                df_data['is_toxic'] = [0] * len(dataset)

            self.df = pd.DataFrame(df_data)

            # Data quality check
            print(f"Data loaded successfully: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            print(f"Label distribution:\n{self.df['is_toxic'].value_counts()}")

            return self.df

        except Exception as e:
            print(f"Error loading from datasets: {e}")
            raise

    def run_complete_pipeline(self):
        """Running complete pipeline"""
        print("=== RUNNING COMPLETE CLASSIFICATION PIPELINE ===")

        self.prepare_data()

        print("\n=== CLASSICAL MODELS ===")
        self.vectorize_text(method='both')
        self.train_classical_models()
        self.evaluate_classical_models()

        print("\n=== HYPERPARAMETER TUNING ===")
        self.hyperparameter_tuning()

        print("\n=== TRANSFORMER MODELS ===")

        print("\n--- Starting RoBERTa training ---")
        roberta_trainer = self.train_roberta()

        print("\n--- Starting CodeBERT training ---")
        codebert_trainer = self.train_codebert()

        report = self.generate_report()

        report.to_csv('model_comparison_report.csv')

        return report

def classifier(dataset_path, model_name):
    """
    Classifier for interface from main.py
    """
    print(f"Running classification with model: {model_name}")

    # Reading cleaned dataset
    excel_dataset_path = dataset_path

    # Initializing classifier
    clf = CodeReviewClassifier(str(excel_dataset_path))

    # Running appropriate pipeline depending on model
    if model_name == 'classic_ml':
        print("=== RUNNING CLASSICAL MODELS ===")
        clf.prepare_data()
        clf.vectorize_text(method='both')
        clf.train_classical_models()
        clf.evaluate_classical_models()
        clf.hyperparameter_tuning()

    elif model_name == 'roberta':
        print("=== RUNNING RoBERTa ===")
        clf.prepare_data()
        clf.train_roberta()

    elif model_name == 'microsoft/codebert-base':
        print("=== RUNNING CodeBERT ===")
        clf.prepare_data()
        clf.train_codebert()

    # Generating report
    report = clf.generate_report()

    return report
