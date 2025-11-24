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


# para stemizar as palavras
from nltk.stem import PorterStemmer

# cria uma instância da classe PorterStemmer
stemmer = PorterStemmer()

# importando biblioteca json
import json
import pickle
import numpy as np

words=[] #lista de palavras-raiz únicas nos dados
classes = [] #lista de tags únicas nos dados
pattern_word_tags_list = [] #lista dos pares de (['palavras', 'da', 'frase'], 'tags')

# palavras a serem ignoradas ao criar o conjunto de dados
ignore_words = ['?', '!',',','.', "'s", "'m"]

# abra o arquivo JSON e carregue os dados dele.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# criando uma função para stemizar as palavras
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)      

    return stem_words


'''
Lista de palavras-tronco classificadas para nosso conjunto de dados:

['todos', 'alg', 'algue', 'sao', 'incrivel', 'ser', 'melhor', 'bluetooth', 'tchau', 'camera', 'pode', 'conversa', 
'legal', 'poderia', 'digito', 'fazer', 'para', 'game', 'adeu', 'ter', 'ouvido', 'ola', 'ajudar', 'ei', 
'oi', 'ola', 'como', 'e', 'depois', 'recente', 'mim', 'mais', 'proximo', 'bom', 'fone', 'favor', 'popular', 
'produto', 'fornecer', 'ver', 'vender', 'mostrar', 'smartphon', 'contar', 'obrigado', 'que', 'o', 'la', 
'ate', 'vez', 'para', 'moda', 'video', 'que', 'qual', 'voce', 'seu']

'''


# criando uma função para criar o corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        # Adicione todos os padrões e tags a uma lista
        for pattern in intent['patterns']:  

            # tokenize o padrão          
            pattern_words = nltk.word_tokenize(pattern)

            # adicione as palavras tokenizadas à lista words
            words.extend(pattern_words)      
                          
            # adicione a "lista de palavras tokenizadas" juntamente com a 'tag' à pattern_word_tags_list
            pattern_word_tags_list.append((pattern_words , intent['tag']))
            
            
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            
    stem_words = get_stem_words(words, ignore_words) 

    # Remova palavras duplicadas de stem_words
    stem_words = set(stem_words)

    # classifique as listas stem_words e classes
    stem_words = sorted(list(stem_words))
    classes = sorted(classes)

    # imprima stem_words
    print('lista stem_words: ' , stem_words)

    return stem_words, classes, pattern_word_tags_list


# Conjunto de Dados de Treinamento: 
# Texto de Entrada ----> como Saco de Palavras 
# Tags-----------------> como Etiqueta

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:
        # exemplo: word_tags = (['Como', 'esta'], 'saudacoes']

        pattern_words = word_tags[0] # ['Como' , 'Esta']
        bag_of_words = []

        # stemizando palavras padrões antes de criar o saco de palavras
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        # Codificação dos dados de entrada 
        for word in stem_words:
            if word in stemmed_pattern_word:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        
        bag.append(bag_of_words)
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Comece com a lista de 0s
        labels_encoding = list([0]*len(classes))  

        # exemplo: word_tags = (['Como', 'esta'], 'saudacoes']

        tag = word_tags[1]   # 'saudacoes'

        tag_index = classes.index(tag)

        # Codificação das etiquetas
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)
        
    return np.array(labels)

def preprocess_train_data():
  
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Converta as palavras-tronco e classes para o formato de arquivo pickel do Python
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(tag_classes, open('classes.pkl','wb'))

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

bow_data  , label_data = preprocess_train_data()

# pós completar o código, remova os comentários das instruções print
print("primeira codificação BOW: " , bow_data[0])
print("primeira codificação de Etiqueta: " , label_data[0])


