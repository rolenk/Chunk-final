import pandas as pd
import pickle
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

loyalty = pd.read_csv("Customer Loyalty History.csv")
flight = pd.read_csv("Customer Flight Activity.csv")

df = pd.merge(
    flight,
    loyalty,
    on="Loyalty Number",
    how="left"
)

df['Churn'] = df['Cancellation Year'].notnull().astype(int)

df['Salary'] = df['Salary'].fillna(df['Salary'].median())

customer_df = df.groupby('Loyalty Number').agg({
    'Total Flights':'sum',
    'Distance':'sum',
    'Points Accumulated':'sum',
    'Points Redeemed':'sum',
    'Salary':'first',
    'CLV':'first',
    'Gender':'first',
    'Education':'first',
    'Marital Status':'first',
    'Loyalty Card':'first',
    'Enrollment Type':'first',
    'Churn':'max'
}).reset_index()

customer_df['Redemption Rate'] = (
    customer_df['Points Redeemed'] /
    (customer_df['Points Accumulated'] + 1)
)

customer_df['Avg Distance Per Flight'] = (
    customer_df['Distance'] /
    (customer_df['Total Flights'] + 1)
)

customer_df['Points Per Flight'] = (
    customer_df['Points Accumulated'] /
    (customer_df['Total Flights'] + 1)
)

customer_df['CLV Per Flight'] = (
    customer_df['CLV'] /
    (customer_df['Total Flights'] + 1)
)

X = customer_df.drop(
    ['Loyalty Number', 'Churn'],
    axis=1
)

y = customer_df['Churn']

ct = ColumnTransformer(
    transformers=[
        (
            'encoder',
            OneHotEncoder(drop='first'),
            [
                'Gender',
                'Education',
                'Marital Status',
                'Loyalty Card',
                'Enrollment Type'
            ]
        )
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

sc = StandardScaler(with_mean=False)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(
    kernel='rbf',
    random_state=0
)

classifier.fit(X_train, y_train)

pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))
pickle.dump(ct, open('encoder.pkl', 'wb'))

print("model saved")