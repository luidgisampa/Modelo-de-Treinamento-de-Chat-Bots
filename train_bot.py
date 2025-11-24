import subprocess
import sys

def install_tensorflow():
    print("Verificando e instalando o TensorFlow...")
    try:
        # Tenta instalar o pacote 'tensorflow' usando pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
        print("TensorFlow instalado com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"Falha ao instalar o TensorFlow: {e}")
        # Se a instalação falhar (provavelmente pelo erro de caminho longo)
        # o script irá parar aqui e informar o usuário sobre o erro.
        print("\nERRO CRÍTICO: A instalação falhou.")
        print("Por favor, habilite o suporte a 'Long Paths' no Windows e reinicie seu PC.")
        sys.exit(1) # Sai do script com um código de erro

# Chama a função de instalação antes de qualquer importação
install_tensorflow()

# Bibliotecas de treinamento do modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile o modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Ajuste e salve o modelo
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('chatbot_model.h5', history)

    print("Modelo Criado e Salvo")


# Chamando os métodos para treinar o modelo
train_x, train_y = preprocess_train_data()

train_bot_model(train_x, train_y)

