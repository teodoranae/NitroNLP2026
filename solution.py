from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("ro_core_news_sm")

def train_classifier(X_train, y_train):
    y_train = (y_train > 0).astype(int)
    model = LGBMClassifier(
        n_estimators=800, 
        learning_rate=0.05, 
        num_leaves=31, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_regressor(X_train, y_train):
    mask = y_train > 0
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    model = LGBMRegressor(
        n_estimators=1000, 
        learning_rate=0.04, 
        num_leaves=45, 
        objective='mae', 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def preprocess_data(df, is_train=True, train_stats=None):
    df['word'] = df['word'].astype(str)
    df['word_lower'] = df['word'].str.lower()
    
    df['len'] = df['word'].apply(len)
    df['is_punctuation'] = df['word'].apply(lambda x: 1 if not x.isalnum() else 0)
    df['is_upper'] = df['word'].apply(lambda x: 1 if x[0].isupper() else 0)
    df['is_title'] = df['word'].apply(lambda x: 1 if x.istitle() else 0)
    df['is_numeric'] = df['word'].apply(lambda x: 1 if x.isdigit() else 0)
    
    df['word_index'] = df['word_id'].apply(lambda x: int(str(x).split('_')[-1]) if '_' in str(x) else 0)
    df['word_position_rev'] = df.groupby(['participant_id', 'text']).cumcount(ascending=False)
    
    if is_train:
        train_stats = {
            'word_freq_map': df['word_lower'].value_counts().to_dict(),
            'pos_cache': {w: (nlp(str(w))[0].pos_, nlp(str(w))[0].dep_, nlp(str(w))[0].is_stop) for w in df['word'].unique()}
        }
        
    df['word_freq'] = df['word_lower'].map(train_stats['word_freq_map']).fillna(1)
    
    vowels = set("aeiouăîâ")
    df['n_vowels'] = df['word_lower'].apply(lambda x: sum(1 for c in x if c in vowels))
    df['vowel_ratio'] = df['n_vowels'] / (df['len'] + 1e-6)
    
    if not is_train:
        new_words = set(df['word'].unique()) - set(train_stats['pos_cache'].keys())
        for w in new_words:
            t = nlp(str(w))[0]
            train_stats['pos_cache'][w] = (t.pos_, t.dep_, t.is_stop)

    df['pos'] = df['word'].map(lambda x: train_stats['pos_cache'][x][0]).astype('category')
    df['dep'] = df['word'].map(lambda x: train_stats['pos_cache'][x][1]).astype('category')
    df['is_stop'] = df['word'].map(lambda x: 1 if train_stats['pos_cache'][x][2] else 0)
    
    for shift_val in [-1, 1]:
        df[f'shift_{shift_val}_len'] = df.groupby(['participant_id', 'text'])['len'].shift(shift_val).fillna(0)
        df[f'shift_{shift_val}_freq'] = df.groupby(['participant_id', 'text'])['word_freq'].shift(shift_val).fillna(0)

    df['text_avg_word_len'] = df.groupby('text')['len'].transform('mean')
    df['text_avg_word_freq'] = df.groupby('text')['word_freq'].transform('mean')

    return df, train_stats

if __name__ == "__main__":
    train = pd.read_csv('train_data.csv')

    train, train_stats = preprocess_data(train, is_train=True)

    features = [
        'len', 'is_punctuation', 'is_upper', 'is_title', 'is_numeric',
        'word_index', 'word_position_rev', 'word_freq',
        'n_vowels', 'vowel_ratio', 'pos', 'dep', 'is_stop',
        'shift_-1_len', 'shift_1_len',
        'shift_-1_freq', 'shift_1_freq',
        'text_avg_word_len', 'text_avg_word_freq'
    ]

    X_train_full = train[features]
    y_train_full = train['answer']

    classifier = train_classifier(X_train_full, y_train_full)
    regressor = train_regressor(X_train_full, y_train_full)

    test = pd.read_csv('test_data.csv')

    test, _ = preprocess_data(test, is_train=False, train_stats=train_stats)
    X_test = test[features]

    test_probs = classifier.predict_proba(X_test)[:, 1]
    test_regressor_values = regressor.predict(X_test)
    
    test_preds = test_probs * test_regressor_values
    test_preds = np.clip(test_preds, 0, None)

    submission = pd.DataFrame({
        'subtaskID': 1,
        'datapointID': test['datapointID'],
        'answer': test_preds
    })

    submission.to_csv('submission.csv', index=False)
