import spacy
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import time
import itertools
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

def preprocess_text(text):
    texto_limpio = re.sub(r"[^\w\s]", "", text)
    # Tokenizamos el texto en palabras
    palabras = word_tokenize(texto_limpio)
    # Obtenemos las stop words en el idioma español
    stop_words = set(stopwords.words("spanish"))
    stop_words.remove("no")
    # Eliminamos las stop words del texto
    palabras_sin_stopwords = [
        palabra for palabra in palabras if palabra.lower() not in stop_words
    ]
    palabras_sin_stopwords = " ".join(palabras_sin_stopwords)

    nlp = spacy.load("es_core_news_lg")
    doc = nlp(palabras_sin_stopwords)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct]
    preprocessed_text = " ".join(tokens)
    features = {}
    doc = nlp(preprocessed_text)
    for token in doc:
        if not token.is_punct:
            if token.lemma_.lower() in features:
                features[token.lemma_.lower()] += 1
            else:
                features[token.lemma_.lower()] = 1
    return features

# def get_features(text):
#     features = {}
#     nlp = spacy.load("es_core_news_lg")
#     doc = nlp(text)
#     for token in doc:
#         if not token.is_punct:
#             if token.lemma_.lower() in features:
#                 features[token.lemma_.lower()] += 1
#             else:
#                 features[token.lemma_.lower()] = 1
#     return features

def analyze_data(respuesta):
    vectorizer_name = "vectorizer3.joblib"
    vectorizer = joblib.load(vectorizer_name)
    model_name = "sentimientos3.sav"
    sentiment_analysis = joblib.load(model_name)
    try:
        features = preprocess_text(str(respuesta))
        X_test = vectorizer.transform([features])
        sentiment = sentiment_analysis.predict(X_test)
        return sentiment[0]
    except TypeError:
        print(respuesta)
        return "none"
    

def analyze_data_gpu(respuestas, nlp, stop_words):
    counter = 0
    resultados = []
    vectorizer_name = "vectorizer5.joblib"
    vectorizer = joblib.load(vectorizer_name)
    model_name = "sentimientos5.sav"
    sentiment_analysis = joblib.load(model_name)
    for respuesta in respuestas:
        try:
            #PREPROCESS TEXT
            texto_limpio = re.sub(r"[^\w\s]", "", str(respuesta))
            # Tokenizamos el texto en palabras
            palabras = word_tokenize(texto_limpio)
            # Eliminamos las stop words del texto
            palabras_sin_stopwords = [
                palabra for palabra in palabras if palabra.lower() not in stop_words
            ]
            palabras_sin_stopwords = " ".join(palabras_sin_stopwords)
            doc = nlp(palabras_sin_stopwords)
            tokens = [token.lemma_.lower() for token in doc if not token.is_punct]
            preprocessed_text = " ".join(tokens)
            #GET FEATURES
            features = {}
            doc = nlp(preprocessed_text)
            for token in doc:
                if not token.is_punct:
                    if token.lemma_.lower() in features:
                        features[token.lemma_.lower()] += 1
                    else:
                        features[token.lemma_.lower()] = 1
            X_test = vectorizer.transform([features])
            #SENTIMENT ANALYSIS
            sentiment = sentiment_analysis.predict(X_test)
            resultados.append(sentiment[0])
        except TypeError:
            print(respuesta)
            resultados.append("none")
        counter += 1
        if counter % 500 == 0:
                print(f"{counter} respuestas analizadas")
    return resultados   
    

if __name__ == "__main__":
    spell = SpellChecker(language="es")
    spell.word_frequency.load_text_file("Spanish.dic")
    spell.word_frequency.load_text_file("new_words.txt")
    # Obtenemos las stop words en el idioma español
    stop_words = set(stopwords.words("spanish"))
    stop_words.remove("no")
    excel_path = r"IDEA202410 Completo.xlsx"
    df = pd.read_excel(excel_path)
    respuestas = df["Respuesta"].tolist()
    identificadores = df["CodUnion2"].tolist()
    preguntas = df["Pregunta"].tolist()
    corrected_reviews = []
    print("Correcting words")
    start = time.time()
    with ProcessPoolExecutor(max_workers=6) as executor:
        for result in executor.map(get_corrected_review, respuestas, itertools.repeat(spell)):
            corrected_reviews.append(result)
    print(f"Words corrected in: {time.time() - start}")
    resultados = []
    print("Analyzing with GPU")
    start = time.time()
    spacy.require_gpu()
    nlp = spacy.load("es_core_news_lg")
    resultados = analyze_data_gpu(corrected_reviews, nlp, stop_words)
    print(f"Answers analyzed in: {time.time() - start}")
    print("Writing to file")
    res_df = pd.DataFrame()
    res_df["CodUnion2"] = identificadores
    res_df["Pregunta"] = preguntas
    res_df["Respuesta"] = respuestas
    res_df["Resultado"] = resultados
    res_df.to_excel("Resultados_IDEA202410_completo.xlsx", index=False)
    print("Done")