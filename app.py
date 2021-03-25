# import all the libraries
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

#inialize flask
app = Flask(__name__ , template_folder = 'template')
model = pickle.load(open('loan.pkl','rb'))
#define the html file to get the user input
@app.route('/')
def home():
    return render_template('index.html')

# prediction function
@app.route('/predict' , methods =['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return  render_template("index.html", prediction_text= "PROBABILITY THAT YOUR LOAN WILL BE APPROVED IS ; {}".format(output))

# output page and logic
@app.route('/results', methods = ['POST'])
def results():
    
    data = request.get_json(force = True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

# main function
if __name__ == "__main__":
    app.run(debug = True)