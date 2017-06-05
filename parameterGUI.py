from modelloader import ModelLoader

class ParameterGUI(object):

    def beginAnlysis(self, business,year,month,day):
        # print "AQUI PONER EL LLAMADO A SU CLASE "

        # Valor de la Prediccion Obtenido
        value_prediction = self.predict_1(business)
        return value_prediction

    def predict_1(self, stock='AAPL'):
		predictor_name = 'predictor_1_'
		loader = ModelLoader()
		predictor = loader.load_predictor(predictor_name, stock, 'NaivePredictor')

		prediction = predictor.predict()
		print "El precio de Cierre es:", prediction[0]
		print "El Limite Inferior es:", prediction[1]
		print "El Limite Superior es:", prediction[2]
		return prediction