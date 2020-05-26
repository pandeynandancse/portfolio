
# # from flask import Blueprint


# # ml_blueprint_object = Blueprint('ml', __name__)












# from flask import Flask
# from flask import Flask, render_template
# import pickle
# app = Flask(__name__)


# model = pickle.load(open('mlmodels/model.pkl', 'rb'))
# @app.route('/mlmodel')
# def mlmodel():
#     return render_template('mlmodel.html')




# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('mlmodel.html', prediction_text='Employee Salary should be $ {}'.format(output))






# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)
