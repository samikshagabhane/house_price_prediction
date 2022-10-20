#import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.linear_model import LinearRegression
import os


house_data = pd.read_csv(r'./data/train.csv')

X_train, X_test, y_train, y_test = train_test_split(house_data.drop(['MedHouseVal'],axis=1), 
                                                    house_data['MedHouseVal'], test_size=0.10, 
                                                    random_state=101)



model = LinearRegression()

model.fit(X_train, y_train)

print("creating pkl file")
print(os.getcwd())
pkl.dump(model,open('./model.pkl','wb'))

pred = model.predict(X_test)


print(pred)