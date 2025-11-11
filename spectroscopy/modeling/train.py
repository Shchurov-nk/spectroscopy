# Code to train models here
import pandas as pd
import tensorflow as tf



def fit_my_model(X_trn, y_trn, X_vld, y_vld, params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1,  activation='linear'))
    
    adam = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model.compile(optimizer=adam, loss='mean_squared_error')

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=params['min_delta'],
        patience=params['patience'],
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True,
    )

    model.fit(X_trn, y_trn, 
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=0,
              callbacks=callback,
              validation_data=(X_vld, y_vld),
             )
    return model