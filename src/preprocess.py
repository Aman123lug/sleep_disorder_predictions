from loggerr import logger
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

STAGE_NAME  = "Data Preprocessing"

class Datapreprocess:
    def __init__(self) -> None:
        pass
    def preprocess_data(self, data):
        data.dropna(inplace=True)
        # data['Sleep Disorder'].fillna('None', inplace=True)
        data.drop('Person ID', axis=1, inplace=True)
                #spliting the blood pressure into two columns
        data['systolic_bp'] = data['Blood Pressure'].apply(lambda x: x.split('/')[0])
        data['diastolic_bp'] = data['Blood Pressure'].apply(lambda x: x.split('/')[1])
        #droping the blood pressure column
        data.drop('Blood Pressure', axis=1, inplace=True)
        data['BMI Category'] = data['BMI Category'].replace('Normal Weight', 'Normal')
              
        label_encoder = preprocessing.LabelEncoder()
        data["Gender"] = label_encoder.fit_transform(data["Gender"])
        data["Occupation"] = label_encoder.fit_transform(data["Occupation"])
        data["BMI Category"] = label_encoder.fit_transform(data["BMI Category"])
        data["Sleep Disorder"] = label_encoder.fit_transform(data["Sleep Disorder"])
        
        st = StandardScaler()
        df["Daily Steps"] = st.fit_transform(df[["Daily Steps"]])
        
        preprocess_data = data
        print(len(preprocess_data.columns))
        path = "data/preprocess_data.csv"
        preprocess_data.to_csv("data/preprocess_data.csv")
        
        return path
        

if __name__ == "__main__":

    try:
        df = pd.read_csv("data/sleep-dataset.csv")
        logger.info(f" >>>> stage {STAGE_NAME} <<<< started !")
        obj = Datapreprocess()
        preprocess_data = obj.preprocess_data(df)
        
        logger.info(f" >>>> stage {STAGE_NAME} <<<< Completed ! \n\n x==================x")
        
    except Exception as e:
        logger.exception(e)
        raise e