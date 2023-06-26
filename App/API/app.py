from flask import Flask, jsonify, request,render_template,send_file
from utilities import predict_pipeline
import pandas as pd
import pickle

with open('Models\pipeline.pickle','rb') as f:
    loaded_pipe=pickle.load(f)


app = Flask(__name__)



@app.route('/template', methods=['GET'])
def download_template():
    template_path = "Templatedata/Template.csv"  # Replace with the actual path to your template CSV file
    return send_file(template_path, as_attachment=True)


@app.route('/',methods=['GET'])
def random():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    clientsfile=request.files['clientsfile']
    client_path = "Clients/"+clientsfile.filename
    clientsfile.save(client_path)

    df = pd.read_csv(client_path)
    columns_to_remove=['CustomerId','Surname']
    df=df.drop(columns=columns_to_remove, axis=1)

    pred = loaded_pipe.predict(df)
    pred_to_label = {0: 'Will not exit', 1: 'Will Exit'}

    # Make a list of text with sentiment.
    data = []
    for pred in pred:
        data.append(( pred, pred_to_label[pred]))


    return render_template('index.html',prediction=data)

if __name__=='__main__':
    app.run(port=3000,debug = True)