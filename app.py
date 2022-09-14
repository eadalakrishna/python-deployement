from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('classifier.pkl')
onehot = joblib.load('OneHotee.joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	print(int_features)
	c = ["FUEL","SIZE","DISTANCE","DESIBEL","AIRFLOW","FREQUENCY"]
	df = pd.DataFrame(int_features,columns=c)	
	l = onehot.transform(df.iloc[:,0:1])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,1:]
	final =pd.concat([l2,t],axis=1)
	

	print(final)
	result = model.predict(final)
	if result == 0:
		pub="indicates the non-extinction state"
	else:
		pub="indicates the extinction state"



	

	return render_template("main.html",prediction_text="Acoustic Extinguisher Fire Status : {}".format(pub))


if __name__ == "__main__":
	app.debug=True
	app.run(host = '127.0.0.1', port =8000)