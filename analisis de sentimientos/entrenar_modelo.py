import spacy
import time
import itertools
import re
import pandas as pd
import joblib
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from concurrent.futures import ProcessPoolExecutor
from spellchecker import SpellChecker


def correct_words(w, spell):
    new_word = spell.correction(w)
    if new_word == None:
        new_word = w
    return new_word

def get_corrected_review(r, spell):
    r = re.sub(r"[^\w\s]", " ", str(r))
    r = str(r).strip()
    r_list = r.split(" ")
    corrected_words = []
    for w in r_list:
        corrected_words.append(correct_words(w, spell))
    new_sentence = " ".join(corrected_words)
    return new_sentence


def preprocess_text(data, nlp, stop_words):
    preprocessed_training_data = []
    for text, label in data:
        texto_limpio = re.sub(r'[^\w\s]', '', text)

        # Tokenizamos el texto en palabras
        palabras = word_tokenize(texto_limpio)

        # Eliminamos las stop words del texto
        palabras_sin_stopwords = [palabra for palabra in palabras if palabra.lower() not in stop_words]
        palabras_sin_stopwords = ' '.join(palabras_sin_stopwords)

        doc = nlp(palabras_sin_stopwords)

        tokens = [token.lemma_.lower() for token in doc if not token.is_punct]

        preprocessed_text = ' '.join(tokens)
        new_data = (preprocessed_text, label)
        preprocessed_training_data.append(new_data)
    return preprocessed_training_data

def get_features(data, nlp):
    training_features = []
    for text, _ in data:
        features = {}
        doc = nlp(text)
        for token in doc:
            if not token.is_punct:
                if token.lemma_.lower() in features:
                    features[token.lemma_.lower()] += 1
                else:
                    features[token.lemma_.lower()] = 1
        training_features.append(features)
    return training_features




if __name__ == "__main__":
    #Inicializamos el spellchecker para corregir faltas de ortografía
    spell = SpellChecker(language="es")
    spell.word_frequency.load_text_file("Spanish.dic")
    spell.word_frequency.load_text_file("new_words.txt")
    # Obtenemos las stop words en el idioma español
    stop_words = set(stopwords.words('spanish'))
    stop_words.remove('no')
    #Leemos el excel en un dataframe
    df = pd.read_excel('test_data2.xlsx')
    df = df.drop_duplicates()
    print(df)
    reseñas = df['Respuesta'].tolist()
    sentimientos = df['Resultado'].tolist()
    training_data = []
    #Corregimos las faltas ortograficas en cada reseña
    reseñas_corregidas = []
    print("Correcting words")
    start = time.time()
    with ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(get_corrected_review, reseñas, itertools.repeat(spell)):
            reseñas_corregidas.append(result)
    print(f"Total time: {time.time() - start}")
    print("Appending data")
    start_time = time.time()
    for reseña, sentimiento in zip(reseñas_corregidas, sentimientos):
        training_data.append((str(reseña), sentimiento))
    print(f"Total time: {time.time()-start_time}")
    spacy.require_gpu()
    nlp = spacy.load("es_core_news_lg")
    print("Preprocessing data")
    start_time = time.time()
    counter = 0
    preprocessed_training_data = preprocess_text(training_data, nlp, stop_words)
    # for result in executor.map(preprocess_text, training_data):
    #     preprocessed_training_data.append(result)
    #     counter += 1
    #     if counter % 10 == 0:
    #         print(counter)
    print(f"Total time: {time.time()-start_time}")
    training_features = []
    print("Getting features")
    start_time = time.time()
    training_features = get_features(preprocessed_training_data, nlp)
    # counter = 0
    # for result in executor.map(get_features, preprocessed_training_data):
    #     training_features.append(result)
    #     counter += 1
    #     if counter % 10 == 0:
    #         print(counter)
    print(f"Total time: {time.time()-start_time}")
    vectorizer = DictVectorizer(sparse=False)

    print('Training vectorizer')
    start_time = time.time()
    X_train = vectorizer.fit_transform(training_features)

    y_train = [label for _, label in preprocessed_training_data]

    vectorizer_name = 'vectorizer5.joblib'
    joblib.dump(vectorizer, vectorizer_name)
    print(f"Total time: {time.time()-start_time}")

    print('Training classifier')
    start_time = time.time()
    classifier = MultinomialNB()
    _ = classifier.fit(X_train, y_train)

    model_name = 'sentimientos5.sav'
    joblib.dump(classifier, model_name)
    print(f"Total time: {time.time()-start_time}")