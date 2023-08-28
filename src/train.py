from loggerr import logger
# from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import pandas as pd
from pathlib import Path
import mlflow
from urllib.parse import urlparse
from mlflow.models import infer_signature
import mlflow.sklearn
import pickle


STAGE_NAME  = "Training"

class Training:
    def _init_(self) -> None:
        pass
    def training(self, path:Path):
        preprocess_data = pd.read_csv(path)
        print(preprocess_data.columns)
        preprocess_data.drop("Unnamed: 0", axis=1, inplace=True)
        print(preprocess_data.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(preprocess_data.drop('Sleep Disorder',axis=1), preprocess_data['Sleep Disorder'], test_size=0.2, random_state=42)
        with mlflow.start_run():
            dtree = DecisionTreeClassifier(max_depth=3)
            
            dtree.fit(X_train, y_train)
            
            training_accuracy = dtree.score(X_train,y_train)
            print("Training Accuracy:", training_accuracy)
            
            y_pred = dtree.predict(X_test)
            print("Testing Accuracy:", accuracy_score(y_pred, y_test))
            test_accracy = accuracy_score(y_pred, y_test)
            
            report = classification_report(y_test, y_pred)
            print("classification report:", report)
            logger.info(f" training successfull !")
            
            mlflow.log_metric("training Score", training_accuracy)
            mlflow.log_metric("test accuracy", test_accracy)
            # mlflow.log_metric("Classification Report", str(report))

            signature = infer_signature(X_train, y_pred)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logger.info(f" MLflow log_metric saved !")

            # Model registry does not work with file store
            # if tracking_url_type_store != "file":
            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the use case,
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #     mlflow.sklearn.log_model(
            #         dtree, r"model/model.pkl", registered_model_name="DecisionTree", signature=signature
            #     )
            # else:
            #     mlflow.sklearn.log_model(dtree, r"model/model.pkl", signature=signature)
            
            with open("metrics.txt", "w+") as f:
                f.write(str(test_accracy))
                logger.info(f" metrics saved !")
            
        # save model
            pickle.dump(dtree, open(r"model/model.pkl", "wb"))
            logger.info(f" model saved !")
            # IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
            # cm=ConfusionMatrixDisplay.from_estimator(IC, y_pred, y_test, normalize='true',  values_format='.2%')
    
            # cm.figure_.savefig('images/confusion_matrix.png')

        
        
if __name__ == "__main__":
    preprocess_path =r"data/preprocess_data.csv"
    
    try:
        logger.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj3 = Training()
        obj3.training(preprocess_path)
        logger.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logger.exception(e)
        raise e
