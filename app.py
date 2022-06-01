import pandas as pd
from flask import Flask, jsonify, request
import pickle
# Code from Best Pipeline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 1.0
exported_pipeline = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, max_features=0.9000000000000001, min_samples_leaf=17, min_samples_split=7, n_estimators=100, subsample=0.55)

exported_pipeline.fit(training_features, training_target)


# Flask app script
#app
app = Flask(__name__)
#routes
@app.route('/', methods=['POST'])
def predict():
    #get data
    
    data = request.get_json(force=True)
    
    #convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    #predictions
    
    try:
      with open('encoder.pkl', 'rb') as f:
          encoder = pickle.load(f)
      data_df = encoder.transform(data_df)
    except:
      print("No encoder exist")
      
    result = exported_pipeline.predict(data_df)
   
    #decode the output
    try:
      with open('target_encoder.pkl', 'rb') as f:
          target_encoder = pickle.load(f)
      result = target_encoder.inverse_transform(result)
    except:
        print("No target encoder exist")
    
    #send back to browser
    output = {'results': result[0]}
    
    #return data
    return jsonify(results=output)
    
if __name__ == "__main__":
    # app.run(debug = True)
    app.run(host ='0.0.0.0', port = 8080, debug = True)
    