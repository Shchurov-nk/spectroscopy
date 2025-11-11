# Code to train models here
import pandas as pd
import tensorflow as tf

class ModelTrainer:
    def __init__(self, X_trn, y_trn, X_vld, y_vld, model_params, ):
        self.X_trn = X_trn
        self.y_trn = y_trn
        self.X_vld = X_vld
        self.y_vld = y_vld
        self.params = model_params
        self.model_path = config
        pass

    def fit_my_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1,  activation='linear'))
        
        adam = tf.keras.optimizers.Adam(learning_rate=self.params['lr'])
        model.compile(optimizer=adam, loss='mean_squared_error')

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=self.params['min_delta'],
            patience=self.params['patience'],
            verbose=0,
            mode='min',
            baseline=None,
            restore_best_weights=True,
        )

        model.fit(
            self.X_trn, self.y_trn, 
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            verbose=0,
            callbacks=callback,
            validation_data=(self.X_vld, self.y_vld),
            )
        
        model_name = nn_models_path / f'NN_.keras'
        model.save(model_name)
        return model