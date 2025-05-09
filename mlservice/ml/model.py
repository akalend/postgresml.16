import json
import psycopg
import pandas as pd
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split

DATABASE_URL = "postgresql://postgres@localhost/test"

"""
A generic model class.

"""
class Model:

	def __init__(self):
		self.cnn = psycopg.connect("dbname=test user=postgres")
		self.query = ''
		self.data = None

	def getQuery(self, numder):
		self.numder = numder
		with self.cnn.cursor() as cur:
			cur.execute("SELECT query, args,name FROM ml_model WHERE sid=%s", [numder])
			row = cur.fetchone();
			if row:
				self.query = row[0]
				self.args = json.loads(row[1])
				self.name = row[2]
				print('Query from {} Ok'.format(numder))
				return True
			else:
				print('row {} not found'.format(numder))
				return False

	def getData(self):
		sql =  "SELECT {}".format(self.query)
		self.data = pd.read_sql(sql, self.cnn) 


	def process(self):
		split = 0.2
		data = self.data
		target = self.args['target'];


		
		if self.args.get('ignored'):
			droplist = self.args['ignored']
			data = data.drop(columns=droplist)
			print('columns', data.columns)

			tmp_features = data.columns[data.dtypes == 'object'].tolist()
			cat_features = [i for i in tmp_features if i not in droplist]
		else:
			cat_features = data.columns[data.dtypes == 'object'].tolist()

		train, test = train_test_split(data, test_size=split,  shuffle=False) 

		X_train, y_train = train.drop(target, axis=1),train[target]
		X_test, y_test   = test.drop(target, axis=1),test[target]

		train = train.drop(target, axis=1)

		pool = Pool(X_train, y_train, cat_features=cat_features)
	
		model = CatBoostClassifier(allow_writing_files=False, task_type="CPU")
		model.fit(pool)
		acc = model.score(X_test, y_test)
		columns = X_train.columns.tolist()

		
		filename = '/tmp/{}.cbm'.format(self.numder)
		model.save_model(filename)
		with open(filename, 'rb') as f:
			binmodel = f.read()

		with self.cnn.cursor() as cur:			
			cur.execute("UPDATE ml_model SET acc=%s,sid=NULL,fieldlist=%s,data=%s,model_type='C'  WHERE name=%s",\
				[acc, str(columns).replace("'",'"'),binmodel,self.name])
			self.cnn.commit()
		
		self.cnn.close()