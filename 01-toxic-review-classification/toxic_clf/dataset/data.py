import pandas as pd
import re
import numpy as np
from collections import defaultdict


class CodeReviewDataPreprocessor:
    def __init__(self):
        self.contraction_map = {"don't": "do not", "doesn't": "does not", "isn't": "is not", "aren't": "are not",
            "can't": "cannot", "couldn't": "could not", "won't": "will not", "we're": "we are", "i'm": "i am"}

        # CORRECTED obscene words dictionary
        self.obscene_words_patterns = self._load_obscene_words()

    def _load_obscene_words(self):
        """
        Load and normalize obscene words dictionary
        """
        # Original data (example from your description)
        raw_obscene_dict = {
            ' ass ': ['[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\\$\\$', '[^a-z]anus', ' a\\*s\\*s', '[^a-z]ass[^a-z ]', 'a[@#\\$%\\^&\\*][@#\\$%\\^&\\*]', '[^a-z]anal ', 'a s s'],

            ' ass hole ': [' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\\$\\$hole'],

            ' bastard ': ['ba[s|z]+t[e|a]+rd'],

            ' bitch ': ['bitches', ' b[w]*i[t]*ch', ' b!tch', ' bi\\+ch', ' b!\\+ch', ' (b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)', ' biatch', ' bi\\*\\*h', ' bytch', 'b i t c h'],

            ' bull shit ': ['bullsh\\*t', 'bull\\$hit', 'bull sh.t'],

            ' cock ': ['[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk', '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'],

            ' crap ': [' (c)(r|[^a-z0-9 ])(a|[^a-z0-9 ])(p|[^a-z0-9 ])([^ ])*', ' (c)([^a-z]*)(r)([^a-z]*)(a)([^a-z]*)(p)', ' c[!@#\\$%\\^\\&\\*]*r[!@#\\$%\\^&\\*]*p', 'cr@p', ' c r a p'],

            ' cunt ': ['cunt', 'c u n t'],

            ' dick ': [' dick[^aeiou]', 'd i c k'],

            ' dumb ': ['(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'],

            ' dumb ass': ['dumbass', 'dubass'],

            ' ass head': ['butthead'],

            ' faggot ': ['faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit', '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot'],

            ' fuck ': ['(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*', '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)', 'f u u c', '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\\*', 'feck ', ' fux ', 'f\\*\\*', 'f\\-ing', 'f\\.u\\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'],

            ' gay ': ['gay', 'homo'],

            ' haha ': ['ha\\*\\*\\*ha'],

            ' idiot ': ['i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots' 'i d i o t'],

            ' jerk ': ['jerk'],

            ' mother fucker': [' motha f', ' mother f', 'motherucker', ' mofo', ' mf '],

            ' nigger ': ['nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'],

            ' pussy ': ['pussy[^c]', 'pusy', 'pussi[^l]', 'pusses', '(p)(u|[^a-z0-9 ])(s|[^a-z0-9 ])(s|[^a-z0-9 ])(y)'],

            ' rape ': ['raped'],

            ' retard ': ['returd', 'retad', 'retard', 'wiktard', 'wikitud'],

            ' sex ': ['sexy', 's3x', 'sexuality'],

            ' shit ': ['shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\\$hit', 's h i t', 'sh\\*tty', 'sh\\*ty', 'sh\\*t'],

            ' shit hole ': ['shythole', 'sh.thole'],

            ' suck ': ['sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'],

            ' whore ': ['wh\\*\\*\\*', 'w h o r e'],

            ' transgender': ['transgender'],

            ' for your fucking information': [' fyfi', '^fyfi'],

            ' get the fuck off': ['gtfo', '^gtfo'],

            ' oh my fucking god ': [' omfg', '^omfg'],

            ' shut the fuck up': [' stfu' '^stfu'],

            ' son of bitch ': [' sob ', '^sob '],

            ' what the fuck ': [' wtf', '^wtf'],

            ' what the hell ': [' wth', '^wth']
        }

        # Dictionary normalization
        normalized_dict = {}
        for word, patterns in raw_obscene_dict.items():
            # Remove spaces in keys
            clean_word = word.strip()

            normalized_patterns = []
            for pattern in patterns:
                # If pattern is a simple word, add word boundaries
                if self._is_simple_word(pattern):
                    normalized_pattern = r'\b' + re.escape(pattern) + r'\b'
                else:
                    # For regex patterns, add word boundaries if not present
                    if not pattern.startswith('\\b') and not pattern.startswith('(?<!'):
                        normalized_pattern = r'\b' + pattern + r'\b'
                    else:
                        normalized_pattern = pattern

                normalized_patterns.append(normalized_pattern)

            normalized_dict[clean_word] = normalized_patterns

        return normalized_dict

    def _is_simple_word(self, text):
        """Check if text is a simple word (without regex special characters)"""
        regex_chars = r'[]().*+?^${}|\\'
        return not any(char in text for char in regex_chars)

    def _safe_regex_replace(self, text, pattern, replacement):
        """
        Safe replacement with regex error handling
        """
        try:
            return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        except re.error as e:
            print(f"Error in regex pattern: {pattern} - {e}")
            return text

    def correct_obscene_words(self, text):
        """Obscene words correction with improved logic"""
        if not isinstance(text, str):
            return ""

        for replacement_word, patterns in self.obscene_words_patterns.items():
            for pattern in patterns:
                text = self._safe_regex_replace(text, pattern, replacement_word)

        return text

    def load_excel_data(self, filepath, sheet_name=0, text_column='text', label_column=None):
        """Load data from Excel file"""
        try:
            self.df = pd.read_excel(filepath, sheet_name=sheet_name)
            print(f"Successfully loaded Excel file: {filepath}")
            print(f"Data size: {self.df.shape}")

            # Column validation
            if text_column not in self.df.columns:
                available_columns = list(self.df.columns)
                print(f"Column '{text_column}' not found. Available columns: {available_columns}")
                if available_columns:
                    text_column = available_columns[0]
                    print(f"Using first available column: '{text_column}'")

            self.text_column = text_column
            self.label_column = label_column

            print(f"\nFirst 3 rows of data:")
            print(self.df.head(3))

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise

    def explore_obscene_words_distribution(self):
        """
        Analyze obscene words distribution in dataset
        """
        if not hasattr(self, 'df') or self.text_column not in self.df.columns:
            print("Data not loaded")
            return

        print("\nANALYSIS OF OBSCENE WORDS IN DATASET:")

        # Temporary column for analysis
        temp_df = self.df.copy()
        temp_df['contains_obscene'] = False

        for word, patterns in self.obscene_words_patterns.items():
            for pattern in patterns:
                mask = temp_df[self.text_column].str.contains(pattern, case=False, na=False, regex=True)
                temp_df.loc[mask, 'contains_obscene'] = True

        obscene_count = temp_df['contains_obscene'].sum()
        total_count = len(temp_df)

        print(f"Texts with obscene words: {obscene_count}/{total_count} ({obscene_count / total_count * 100:.2f}%)")

        if obscene_count > 0:
            print("\nExamples of texts with obscene words:")
            obscene_samples = temp_df[temp_df['contains_obscene']].head(5)
            for idx, row in obscene_samples.iterrows():
                print(f"- {row[self.text_column][:100]}...")

    def preprocess_text(self, text):
        """Full text preprocessing pipeline"""
        if not isinstance(text, str):
            return ""

        processing_pipeline = [self.remove_urls, self.remove_code_snippets, self.expand_contractions,
            self.remove_repeated_chars, self.correct_obscene_words,  # obscene words correction
            self.remove_special_chars, self.additional_cleaning]

        for func in processing_pipeline:
            text = func(text)

        return text

    # Other methods remain unchanged
    def remove_urls(self, text):
        if not isinstance(text, str):
            return ""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def remove_code_snippets(self, text):
        if not isinstance(text, str):
            return ""
        code_pattern = r'```.*?```|`.*?`'
        return re.sub(code_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    def expand_contractions(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        for contraction, expansion in self.contraction_map.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text

    def remove_repeated_chars(self, text):
        if not isinstance(text, str):
            return ""
        pattern = r'(.)\1{2,}'
        return re.sub(pattern, r'\1\1', text)

    def remove_special_chars(self, text):
        if not isinstance(text, str):
            return ""
        cleaned_text = re.sub(r'[&|#|^|*|@|~|`]', '', text)
        return cleaned_text

    def additional_cleaning(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def clean_data(self, remove_duplicates=True, remove_empty_texts=True):
        initial_size = len(self.df)
        print(f"Initial data size: {initial_size}")

        self.df = self.df.dropna(subset=[self.text_column])
        after_missing = len(self.df)
        print(f"Removed missing values: {initial_size - after_missing}")

        if remove_empty_texts:
            self.df = self.df[self.df[self.text_column].astype(str).str.strip().str.len() > 0]
            after_empty = len(self.df)
            print(f"Removed empty texts: {after_missing - after_empty}")

        if remove_duplicates:
            self.df = self.df.drop_duplicates(subset=[self.text_column])
            after_duplicates = len(self.df)
            print(f"Removed duplicates: {after_empty - after_duplicates}")

        print(f"Final data size: {len(self.df)}")

    def preprocess_dataset(self):
        print("\nStarting text preprocessing...")

        self.df['original_text'] = self.df[self.text_column].copy()
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.preprocess_text)

        initial_size = len(self.df)
        self.df = self.df[self.df['cleaned_text'].str.len() > 0]
        print(f"Removed empty texts after cleaning: {initial_size - len(self.df)}")
