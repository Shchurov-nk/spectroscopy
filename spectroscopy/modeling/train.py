import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
import datetime

y_paths = {
    'y_trn': '../data/processed/y_trn_ions.feather',
    'y_vld': '../data/processed/y_vld_ions.feather',
    'y_tst': '../data/processed/y_tst_ions.feather'
}

raman_training_paths = {
    'X_trn': '../data/processed/X_trn_raman.feather',
    'X_vld': '../data/processed/X_vld_raman.feather',
    'X_tst': '../data/processed/X_tst_raman.feather',
    'mask': '../data/interim/masks/fcbf_raman.feather',
    'name': 'Raman masked'
}

absorp_training_paths = {
    'X_trn': '../data/processed/X_trn_absorption.feather',
    'X_vld': '../data/processed/X_vld_absorption.feather',
    'X_tst': '../data/processed/X_tst_absorption.feather',
    'mask': '../data/interim/masks/fcbf_absorption.feather',
    'name': 'Absorption masked'
}

raman_no_mask_training_paths = {
    'X_trn': '../data/processed/X_trn_raman.feather',
    'X_vld': '../data/processed/X_vld_raman.feather',
    'X_tst': '../data/processed/X_tst_raman.feather',
    'mask': None,
    'name': 'Raman'
}

absorp_no_mask_training_paths = {
    'X_trn': '../data/processed/X_trn_absorption.feather',
    'X_vld': '../data/processed/X_vld_absorption.feather',
    'X_tst': '../data/processed/X_tst_absorption.feather',
    'mask': None,
    'name': 'Absorption'
}

model_params = dict(
    lr = 0.001,
    epochs = 500,
    batch_size = 32,
    patience = 10,
    min_delta = 0
)


def prepare_single_spectrum(training_params):
    X_trn = pd.read_feather(training_params['X_trn'])
    X_vld = pd.read_feather(training_params['X_vld'])
    X_tst = pd.read_feather(training_params['X_tst'])

    mask_path = training_params['mask']
    if mask_path:
        mask = pd.read_feather(mask_path)
        X_trn = X_trn.loc[:, mask['Targets']]
        X_vld = X_vld.loc[:, mask['Targets']]
        X_tst = X_tst.loc[:, mask['Targets']]
    return X_trn, X_vld, X_tst

def prepare_combined_spectrum(raman_training_params, absorp_training_params):
    X_trn_raman, X_vld_raman, X_tst_raman = prepare_single_spectrum(raman_training_params)
    X_trn_absorp, X_vld_absorp, X_tst_absorp = prepare_single_spectrum(absorp_training_params)

    X_trn = pd.concat([X_trn_raman, X_trn_absorp], axis=1)
    X_vld = pd.concat([X_vld_raman, X_vld_absorp], axis=1)
    X_tst = pd.concat([X_tst_raman, X_tst_absorp], axis=1)
    return X_trn, X_vld, X_tst

def apply_scaling(X_trn, y_trn, X_vld, y_vld, X_tst, y_tst):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_trn = pd.DataFrame(scaler_X.fit_transform(X_trn), index=X_trn.index, columns=X_trn.columns)
    y_trn = pd.DataFrame(scaler_y.fit_transform(y_trn), index=y_trn.index, columns=y_trn.columns)
    X_vld = pd.DataFrame(scaler_X.transform(X_vld), index=X_vld.index, columns=X_vld.columns)
    y_vld = pd.DataFrame(scaler_y.transform(y_vld), index=y_vld.index, columns=y_vld.columns)
    X_tst = pd.DataFrame(scaler_X.transform(X_tst), index=X_tst.index, columns=X_tst.columns)
    y_tst = pd.DataFrame(scaler_y.transform(y_tst), index=y_tst.index, columns=y_tst.columns)
    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst, scaler_X, scaler_y

def train_keras_model(X_trn, y_trn, X_vld, y_vld, model_params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(6, activation='linear'))

    adam = tf.keras.optimizers.Adam(learning_rate=model_params['lr'])
    model.compile(optimizer=adam, loss='mean_squared_error')

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=model_params['min_delta'],
        patience=model_params['patience'],
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True,
    )

    model.fit(
        X_trn, y_trn, 
        batch_size=model_params['batch_size'],
        epochs=model_params['epochs'],
        verbose=0,
        callbacks=[callback],
        validation_data=(X_vld, y_vld),
        )
    return model

def get_model_perfomance(model, X, y, scaler_y):
    pred = model.predict(X)
    pred = scaler_y.inverse_transform(pred)
    true = scaler_y.inverse_transform(y)

    mae = mean_absolute_error(true, pred, multioutput='raw_values')
    r2 = r2_score(true, pred, multioutput='raw_values')

    return mae, r2

def main():
    single_spectrum_training = [
        raman_training_paths, 
        absorp_training_paths, 
        raman_no_mask_training_paths, 
        absorp_no_mask_training_paths
        ]

    combined_spectrum_training = [
        (raman_training_paths, absorp_training_paths, 'Combined'),
        (raman_no_mask_training_paths, absorp_no_mask_training_paths, 'Combined masked')
    ]

    all_model_metrics = []

    for spectrum_params in single_spectrum_training:
        X_trn, X_vld, X_tst = prepare_single_spectrum(spectrum_params)

        y_trn = pd.read_feather(y_paths['y_trn'])
        y_vld = pd.read_feather(y_paths['y_vld'])
        y_tst = pd.read_feather(y_paths['y_tst'])

        X_trn, y_trn, X_vld, y_vld, X_tst, y_tst, scaler_X, scaler_y = apply_scaling(X_trn, y_trn, X_vld, y_vld, X_tst, y_tst)
        model = train_keras_model(X_trn, y_trn, X_vld, y_vld, model_params)
        mae, r2 = get_model_perfomance(model, X_tst, y_tst, scaler_y)
        for i, ion_name in enumerate(y_tst.columns):
            all_model_metrics.append((spectrum_params['name'], ion_name, 'MAE', mae[i]))
            all_model_metrics.append((spectrum_params['name'], ion_name, 'R²', r2[i]))

    for raman_params, absorp_params, dataset_name in combined_spectrum_training:
        X_trn, X_vld, X_tst = prepare_combined_spectrum(raman_params, absorp_params)
        
        y_trn = pd.read_feather(y_paths['y_trn'])
        y_vld = pd.read_feather(y_paths['y_vld'])
        y_tst = pd.read_feather(y_paths['y_tst'])

        X_trn, y_trn, X_vld, y_vld, X_tst, y_tst, scaler_X, scaler_y = apply_scaling(X_trn, y_trn, X_vld, y_vld, X_tst, y_tst)
        model = train_keras_model(X_trn, y_trn, X_vld, y_vld, model_params)
        mae, r2 = get_model_perfomance(model, X_tst, y_tst, scaler_y)
        for i, ion_name in enumerate(y_tst.columns):
            all_model_metrics.append((dataset_name, ion_name, 'MAE', mae[i]))
            all_model_metrics.append((dataset_name, ion_name, 'R²', r2[i]))

    result_df = pd.DataFrame(all_model_metrics, columns=['dataset', 'ion', 'metric', 'value'])
    result_df.to_csv('../../reports/model_perfomance_metrics.csv')

if __name__ == '__main__':
    main()