import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

class AnnTraining:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test        
        self.y_test = y_test
        model =self.build_model()
        model =self.compile_model(model)
        self.train_model(model)

    def build_model(self):
        print("Building ANN model...")
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)))  #`HL1` connected with Input Layer
        model.add(Dense(32, activation='relu'))  # `HL2` connected with `HL1`
        model.add(Dense(1, activation='sigmoid'))  # Output Layer   
        print(model.summary())
        return model
    
    def compile_model(self, model):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Model compiled with Adam optimizer and binary crossentropy loss.")
        return model

    def train_model(self, model):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        print("Starting model training...")
        history = model.fit(
            self.X_train, 
            self.y_train, 
            validation_data=(self.X_test, self.y_test),
            epochs=100, 
            callbacks=[tensorboard_callback, early_stopping_callback]
        )  
        model.save('ann_model.h5')
        print("Model training completed and saved as 'ann_model.h5'.") 

    def view_tensorboard(self):
        print("To view TensorBoard, run the following command in your terminal:")
        print("tensorboard --logdir=logs/fit")
        print("Then open your web browser and go to http://localhost:6006")    