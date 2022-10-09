import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json
from urllib.request import urlretrieve

import logging

class TranslationClassifier:

    def __init__(self, model_path):
        logging.info("Translation class initialized")
        self.model = load_model(model_path)

        with open('static/variable.json') as f:
            data = json.load(f)

        self.max_eng = data['max_eng']
        self.eng_vocab = data['eng_vocab']
        self.fr_vocab = data['fr_vocab']

        logging.info("Model is loaded!")
        

    # def helo(self, text: str):
    #     inputList = text.split()
    #     sentence = [[key for word in inputList for key, val in self.eng_vocab.items()  if val == word ]]
    #     sentence = pad_sequences(sentence, maxlen=self.max_eng, padding='post')
    #     predictions = self.model.predict(sentence)
    #     return (' '.join([self.fr_vocab[np.argmax(x)] for x in predictions[0] if np.argmax(x)>0]))

    def testPrint(self, text: str):
        inputList = text.split()
        sentence = [[key for word in inputList for key, val in self.eng_vocab.items()  if val == word ]]
        sentence = pad_sequences(sentence, maxlen=self.max_eng, padding='post')
        predictions = self.model.predict(sentence)
        final_pred = [self.fr_vocab[str(np.argmax(x))] for x in predictions[0] if str(np.argmax(x)) != '0']
        final_pred_string = " ".join(final_pred)
        return (final_pred_string)


# def final_predictions_model2(sentence):
    
#     y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
#     y_id_to_word[0] = '<PAD>'

#     x_id_to_word = {value: key for key, value in eng_tokenizer.word_index.items()}
#     inputList = sentence.split()

#     sentence = [[key for word in inputList for key, val in x_id_to_word.items()  if val == word ]]

#     sentence = pad_sequences(sentence, maxlen=max_eng, padding='post')
#     predictions = ok.predict(sentence)

#     return ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0] if np.argmax(x)>0])