import pandas as pd
import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

#loading model
model = pickle.load(open(r'C:\Users\06564176686\repos\Store-Sales-Prediction\model\model_rossmann.pkl', 'rb'))

# initializa API
app = Flask(__name__)

@app.route('/rossmann/predict', methods = ['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json:
        # unique example
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        # multiple examples
        else:
             test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
                
        # Instantiate Rossmann class
        pipeline = Rossmann()
                
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
            
        # data preparation
        df3 = pipeline.datapreparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
    else:
        return Response('{}', status=200, mimetype='aookication/json')

if __name__ == 'main':
    app.run('0.0.0.0')