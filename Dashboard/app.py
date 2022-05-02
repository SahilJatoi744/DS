from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, feature_descriptions

X_train,y_train,X_test,y_test = titanic_survive()
model = RandomForestClassifier().fit(X_train,y_train)

explainer = ClassifierExplainer(model,X_test,y_test, cats=['Sex','Deck','Embarked'],descriptions=feature_descriptions,labels=['Not survived','Survived'])

ExplainerDashboard(explainer).run()