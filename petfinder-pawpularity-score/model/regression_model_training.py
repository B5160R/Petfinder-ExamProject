import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../data/train.csv')
# print(df.head())

X = df[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']]
y = df['Pawpularity']

reg = LinearRegression()
reg.fit(X, y)

print(reg.coef_)

# save model
import joblib
joblib.dump(reg, 'regression_model.pkl')

# load model
model = joblib.load('regression_model.pkl')

# test model
test = pd.read_csv('../data/test.csv')
print(model.predict(test[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']]))