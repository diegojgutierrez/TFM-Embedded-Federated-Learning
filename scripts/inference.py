import sys
import pandas as pd
import tensorflow as tf
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def drop_columns(df_train):
    many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
    cols_to_drop = list(set(many_null_cols))
    return df_train.drop(cols_to_drop, axis=1)

def evaluate_model(y_true, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return [metrics.precision_score(y_true, y_pred, average = 'macro'), metrics.recall_score(y_true, y_pred, average = 'macro'), metrics.f1_score(y_true, y_pred, average='macro'), metrics.accuracy_score(y_true, y_pred), metrics.auc(fpr, tpr)]

train = pd.read_csv("../data/processed/train2_data_" + sys.argv[1] + ".csv")

train = drop_columns(train)
train['Amount'] = boxcox1p(train['Amount'], boxcox_normmax(train['Amount'] + 1))

X = train.drop("Class", 1)
Y = train["Class"]

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('../models/NeuralNet/my_model.h5')

# Show the model architecture
print(model.summary())

scaler = StandardScaler()
X[:] = scaler.fit_transform(X.values)

y_pred = model.predict(X)
y_pred_2 = y_pred.round()

print(evaluate_model(Y, y_pred_2))


