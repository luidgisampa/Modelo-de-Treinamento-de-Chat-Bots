import nltk
import subprocess
import sys
import os

# ... (Mantenha a função install_package() exatamente como está) ...
def install_package(package_name):
    """Garante que o pacote Python esteja instalado via pip."""
    try:
        __import__(package_name)
        print(f"✅ Pacote '{package_name}' já instalado.")
    except ImportError:
        print(f"📦 Pacote '{package_name}' não encontrado. Instalando via pip...")
        # ... (código de instalação) ...

def download_nltk_resource(resource_name, download_dir=None):
    """Garante que o recurso de dados NLTK esteja baixado, lidando com erros."""
    
    # A maneira mais fácil de verificar a existência é tentar fazer o download em modo silencioso
    # e contar com o NLTK para pular o download se já existir.
    print(f"Verificando/Baixando recurso de dados '{resource_name}'...")

    try:
        # Tenta baixar. Se já existir, o NLTK apenas imprime "Package X is already up-to-date!"
        # Usamos quiet=True para evitar a janela gráfica.
        nltk.download(resource_name, quiet=True, download_dir=download_dir)
        print(f"✅ Recurso '{resource_name}' garantido.")
        
    except Exception as e:
        # Isso capturará a maioria dos erros de rede ou permissão
        print(f"❌ Falha crítica ao baixar o recurso '{resource_name}': {e}")
        sys.exit(1)

# --- EXECUÇÃO PRINCIPAL ---

# 1. Garante que o pacote NLTK esteja instalado via pip
install_package("nltk")

# 2. Garante que os dados necessários estejam baixados
# O erro pedia especificamente por 'tokenizers/punkt_tab/english/'
# O nome do pacote a ser baixado é 'punkt_tab' ou 'punkt'
download_nltk_resource('punkt_tab')
download_nltk_resource('punkt')
download_nltk_resource('wordnet')

print("\nTodos os pacotes e dados NLTK estão prontos para uso.")


# palavras a serem ignoradas/omitidas ao enquadrar o conjunto de dados
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Biblioteca load_model
import tensorflow
from data_preprocessing import get_stem_words

# carregue o modelo
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words)
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    for word in words:            
            if word in input_word_token_2:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
    bag.append(bag_of_words)

    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # extraia a classe de predicted_class_label
   predicted_class = classes[predicted_class_label]

   # agora que temos a tag prevista, selecione uma resposta aleatória

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # selecione uma resposta aleatória do robô
        bot_response = random.choice(intent['responses'])
    
        return bot_response
    

print("Oi, eu sou a Estela, como posso ajudar?")

while True:

    # obtenha a entrada do usuário
    user_input = input('Digite sua mensagem aqui: ')

    response = bot_response(user_input)
    print("Resposta do Robô: ", response)
