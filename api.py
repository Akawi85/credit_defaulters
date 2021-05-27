#!/usr/bin/env python3

# import relevant libraries
from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from joblib import load

# Load model
loaded_model = keras.models.load_model('./model_dir/nn_model.h5')
scaler = load('scaler.joblib')

print('@@ Model Loaded!')

# -------------------------------------------------------------------------------------------------
def predict_defaults(data_ints):
    """ Get Data from Form and Predict Class """

    # convert list of tuples to an array of appropriate shape and then to a dataframe
    query_data = np.array(data_ints).reshape(1, -1)
    
    # scale the data using standard scaler
    scaled_data = scaler.transform(query_data)

    # predict the scaled data
    result = loaded_model.predict(scaled_data)

    print('@@ Scaled Data = ', scaled_data)
    print('@@ Raw result = ', result)
        
    return result

# ---->>> Credit Default Prediction End <<<-----


# api definition
app = Flask(__name__)


# ---------------------------------------------------------------------------------------------------
@app.route("/")
def display_form():
    """ Display the home page """

    return render_template('./index.html')

# ---------------------------------------------------------------------------------------------------
# define the predict function
@app.route('/predict', methods= ['POST']) # endpoint url will contain /predict
def predict():
    """ Display the prediction page """

    if request.method == 'POST':
        data_dict = request.form.to_dict() # convert the form fields to dictionary
        # the dictionary above also returns a value for the predict button as an empty string,
        # we'll subset this dictionary using a condition
        a_subset = {key: value for key, value in data_dict.items() if value != ''}
        data_list = list(a_subset.values()) # convert the values of the dictionary to list

        # get the index of the vaues in the list
        gender = data_list[0]; education = data_list[1]; age = data_list[2]; marital = data_list[3]; loan_amnt = data_list[4]
        bill_amnt_1 = data_list[5]; pay_amnt_1 = data_list[6]; pay_status_1 = data_list[7]
        bill_amnt_2 = data_list[8]; pay_amnt_2 = data_list[9]; pay_status_2 = data_list[10]
        bill_amnt_3 = data_list[11]; pay_amnt_3 = data_list[12]; pay_status_3 = data_list[13]
        bill_amnt_4 = data_list[14]; pay_amnt_4 = data_list[15]; pay_status_4 = data_list[16]
        bill_amnt_5 = data_list[17]; pay_amnt_5 = data_list[18]; pay_status_5 = data_list[19]
        bill_amnt_6 = data_list[20]; pay_amnt_6 = data_list[21]; pay_status_6 = data_list[22]

        # order the values in the format of the training data
        ord_data_list = [loan_amnt, gender, education, marital, age,
                         pay_status_1, pay_status_2, pay_status_3, pay_status_4, pay_status_5, pay_status_6,
                         bill_amnt_1, bill_amnt_2, bill_amnt_3, bill_amnt_4, bill_amnt_5, bill_amnt_6,
                         pay_amnt_1, pay_amnt_2, pay_amnt_3, pay_amnt_4, pay_amnt_5, pay_amnt_6]

        # convert the ordered values to integers
        data_ints = list(map(int, ord_data_list)) 

        print('@@ Raw Data = ', data_ints)
        prediction = predict_defaults(data_ints)

        if prediction >= 0.3: # Adjust the threshold to capture more defaulters due to class imbalance of the training data
            pred = 'Default'
        else:
            pred = 'Not Default'

    return render_template('./result.html', prediction = pred)

# write the main function
if __name__ == '__main__':
    app.run(debug = True)