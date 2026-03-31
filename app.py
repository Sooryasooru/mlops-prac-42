from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def train_model():
    X,y = fetch_california_housing(return_X_y=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    model = LinearRegression()
    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    score = r2_score(y_test,pred)

    return model,score

if __name__ == "__main__":
    model,score =train_model()
    print("r2 score",score)