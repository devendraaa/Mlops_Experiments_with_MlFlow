import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# tracking mflow using http
mlflow.set_tracking_uri("http://127.0.0.1:5000")
#load dataset
wine = load_wine()
x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

#define params for Random forest model
max_dept = 10
n_estimator = 10

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_dept, n_estimators=n_estimator, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_dept)
    mlflow.log_param('n_estimator', n_estimator)

    print("accuracy", accuracy)

    # creating confusion metrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel('Predictions')
    plt.title('Confusion_metrix')

    # save plot
    plt.savefig("confusion_metrix.png")

    #log artifacts using mlflow
    mlflow.log_artifact("confusion_metrix.png")
    mlflow.log_artifact(__file__)

    print(accuracy)
