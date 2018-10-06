import sys
from keras.models import Sequential, save_model
from keras.layers import Dense, Activation
from utils import clean_data

SUBMISSION = sys.argv[1]

TRAINING = clean_data('data/train.csv')

Y_train = TRAINING['Survived'].values
X_train = TRAINING.drop(columns=['Survived'])

def create_model():
    model = Sequential()

    model.add(Dense(64, input_dim=6))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('relu'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model

MODEL = create_model()

MODEL.fit(
    X_train,
    Y_train,
    epochs=10,
    steps_per_epoch=800,
    validation_split=0.2,
    validation_steps=400
)

save_model(MODEL, f"models/{SUBMISSION}.h5")
