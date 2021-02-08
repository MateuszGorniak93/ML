import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#ignorowanie warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


#inicjalizacja dataset iris
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

#zamiana kodowania z "labelowego" na one hot - wartosc 1 dla pozycji wartosci, w przeciwnym wypadku 0
oh_enc = OneHotEncoder()
Y = oh_enc.fit_transform(y[:, np.newaxis]).toarray()

#normalizacja wartosci - wartosci w dataset maja teraz (na potrzeby NN) wartosc z zakresu 0-1, 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#podział dataset na zbior do trenowania i na zbior do weryfikacji treningu
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.5, random_state=2)

n_features = X.shape[1]
n_classes = Y.shape[1]

#funkcja odpowiedzialna za tworzenie modelu
def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        #deklaracja "miejsca na warstwy NN"
        model = Sequential(name=name)
        for i in range(n):
            #dodawanie warstwy do modelu, z funkcja atkywacji relu
            model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
        #dodawanie warstwy wyjsciowej, funkcja aktywacji softmax
        model.add(Dense(output_dim, activation='softmax'))

        #kompilacja modelu z danymi parametrami 
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return model
    return create_model

#utworzenie modelu z n warstwami 
models = [create_custom_model(n_features, n_classes, 8, i, 'model_{}'.format(i)) 
          for i in range(1, 4)]

#info o modelu
for create_model in models:
    create_model().summary()


#miejsce na callbacks
cb_dict = {}

#callback - klasa "kontrolująca" proces treningu 
cb = TensorBoard()

for create_model in models:
    model = create_model()
    print('Model No.:', model.name)
    history_callback = model.fit(X_train, Y_train,batch_size=5,epochs=50,verbose=0,validation_data=(X_test, Y_test),callbacks=[cb])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test: POMYLKA:', score[0])
    print('Test: DOKLADNOSC:', score[1])
    
    cb_dict[model.name] = [history_callback, model]



#funkcja do tworzenia modelu - wymagana przez KerasClassifier
custom_model_1 = create_custom_model(n_features, n_classes, 8, 3)
#tworzenie modelu NN
kearas_model_1 = KerasClassifier(build_fn=custom_model_1, epochs=100, batch_size=5, verbose=0)

#"wynik" NN uzyskany poprzez cross validation
scores = cross_val_score(kearas_model_1, X_scaled, Y, cv=10)
print("DOKLADNOSC : {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std()))