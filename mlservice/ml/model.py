import json
import psycopg
import pandas as pd
from catboost import CatBoostClassifier,Pool,CatBoostRegressor
from sklearn.model_selection import train_test_split

DATABASE_URL = "postgresql://postgres@localhost/test"
LOG = "/tmp/ml2.log"

"""
A generic model class.

"""
class Model:

	def __init__(self, ip):
		self.cnn = psycopg.connect("dbname=postgres user=postgres host={}".format(ip))
		self.query = ''
		self.data = None
		self.type = None
		self.ip = ip
		self.log("start\n")

	def log(self, text):
		pass
		# with open(LOG, 'a') as f:
		# 	f.write(text)


	def getQuery(self, numder):
		self.numder = numder
		self.log("getQuery\n")
		with self.cnn.cursor() as cur:
			cur.execute("SELECT query, args,name,model_type FROM ml_model WHERE sid=%s", [numder])
			row = cur.fetchone();			
			if 1:
				self.query = row[0].decode('ascii')
				self.args = json.loads(row[1])
				self.name = row[2].decode('ascii')
				self.type = row[3].decode('ascii')
				self.log("Ok\n")
				return True
			else:
				self.log("False\n")
				return False

		self.log("False\n")
		return False

	def getData(self):
		self.log("getData\n")
		sql =  "SELECT {}".format(self.query)
		self.data = pd.read_sql(sql, self.cnn) 


	def process(self):
		self.log("process\n")
		split = 0.2
		data = self.data
		target = self.args['target'];

		if self.args.get('ignored'):
			droplist = self.args['ignored']
			data = data.drop(columns=droplist)

			tmp_features = data.columns[data.dtypes == 'object'].tolist()
			cat_features = [i for i in tmp_features if i not in droplist]
		else:
			cat_features = data.columns[data.dtypes == 'object'].tolist()

		data = data.dropna()
		if target in cat_features:
			cat_features.remove(target)

		train, test = train_test_split(data, test_size=split,  shuffle=False) 

		X_train, y_train = train.drop(target, axis=1),train[target]
		X_test, y_test   = test.drop(target, axis=1),test[target]

		train = train.drop(target, axis=1)

		pool = Pool(X_train, y_train, cat_features=cat_features)
	
		out = "type:'{}'".format(self.type)
		if self.type == 'C':
			model = CatBoostClassifier(allow_writing_files=False, task_type="CPU")
		elif self.type == 'R':
			model = CatBoostRegressor(allow_writing_files=False, task_type="CPU")
		else:
			print('****** errr', )
			self.log(out);

		print( 'type', self.type)

		model.fit(pool)
		self.log(out)

		self.acc = model.score(X_test, y_test)
		self.columns = X_train.columns.tolist()
		self.save(model);


	def save(self, model):
		filename = '/tmp/{}.cbm'.format(self.numder)
		model.save_model(filename)
		print('acc', self.acc)
		with open(filename, 'rb') as f:
			binmodel = f.read()

			with self.cnn.cursor() as cur:			
				cur.execute("UPDATE ml_model SET acc=%s,sid=NULL,fieldlist=%s,data=%s,model_type=%s  WHERE name=%s",\
					[self.acc, str(self.columns).replace("'",'"'),binmodel, self.type,self.name])
				self.cnn.commit()
		
		self.cnn.close()