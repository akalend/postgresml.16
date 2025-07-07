from typing import Union
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi import HTTPException, Depends, status, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime as dt
from asgiref.sync import async_to_sync
from starlette.responses import JSONResponse
from passlib.context import CryptContext
from pydantic import BaseModel
from hashlib  import md5
import asyncio, os
from time import sleep, time
import datetime, subprocess
import docker
import jwt
import json
import psycopg2
import random
import redis
import shutil



# Expire session time in sec
EXPIRE_SESSION = 100

ALGORITHM = "HS256"
SECRET_KEY = os.getenv('SECRET_KEY', 'AZ03')
SALT_PSW = os.getenv('SALT_PSW', '23.xZ%')

db_params = {
	"dbname": "users",
	"user": "postgres",
	"password": None,
	"host": "127.0.0.1",
	"port": "5432"
}

app = FastAPI()

class JobItem(BaseModel):
	num: int
	ip: str

class Query(BaseModel):
	query: str
	id: str

class Id(BaseModel):
	id: str

class User(BaseModel):
	name: str
	pswd: str
	email:str


def closeDb(conn,cur):
	if cur:
		cur.close()
	if conn:
		conn.close()

def create_access_token(data: dict):
	encoded_jwt = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
	return encoded_jwt


def verify_token(req: Request):
		token = req.headers.get("X-Token")
		if token is None:
			raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
								detail="Invalid authentication credentials 1")

		try:
			payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
			user: str = payload.get("email")
			if user is None:
				raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
									detail="Invalid authentication credentials 2")
			ts: int = payload.get("time")

			if time()-ts > 3600:
				raise HTTPException(status_code=419, 
					detail="Timeout Authentification")
		except Exception as e:
			print(e)
			raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
								detail="Invalid authentication credentials 3")
		return True

def get_data_from_token(token):
	id = None
	ts = 0

	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		id: str = payload.get("email")
		ts: int = payload.get("time")
	except:
		pass
	return (id, ts)



def checkUser(email, pswd):
	try:
		conn = psycopg2.connect(**db_params)
		cur = conn.cursor()

		print(pswd, email)
		sql = """
		SELECT  count(user) FROM users
			WHERE password=%s AND email=%s
		"""
		cur.execute(sql, (pswd, email))
		res = cur.fetchone()

		if res[0] == 0:
			closeDb(conn, cur)
			return  404

	except psycopg2.Error as e:

		closeDb(conn, cur)
		return  400

	closeDb(conn, cur)
	return 200

def setkey(id):
	r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

	t2 = int(time() + EXPIRE_SESSION)
	print(id, 'expire', t2)
	r.set(id, t2, ex=EXPIRE_SESSION)
	r.quit()


@app.put("/ml", status_code=202)
async def create_model(jobItem: JobItem):
	print('new request',jobItem)
	r = redis.Redis(host='127.0.0.1')
	out = "{}:{}".format( jobItem.ip, jobItem.num)
	print(out)
	r.rpush('mlkey', out)
	r.quit()
	return None


@app.get("/checkcode/{code}/{email}")
async def checkCode(response: Response, code: str, email: str):
	try:
		conn = psycopg2.connect(**db_params)
		cur = conn.cursor()

		print(code, email)
		sql = """
		SELECT  count(user) FROM users
			WHERE code=%s AND email=%s
		"""
		cur.execute(sql, (code, email))
		res = cur.fetchone()

		if res[0] == 0:
			closeDb(conn, cur)
			response.status_code = 404
			return  "code not found"

		# Commit the transaction to save changes to the database

	except psycopg2.Error as e:

		closeDb(conn, cur)
		response.status_code = status.HTTP_400_BAD_REQUEST
		return  {"error": str(e)}


	closeDb(conn, cur)
	return 'Ok'

@app.post("/sigout")
async def sigout(request: Request, response: Response):

	body = await request.body()
	data = json.loads(body.decode('utf-8'))
	db = request.cookies.get('db')
	token = request.cookies.get('token')

	if (db is not None) and (token is not None):		
		user,ts = get_data_from_token(token)
		print(1, user,data['email'],ts, db)
		if data['email'] == user:
			print(data['email'], user)
			client = docker.from_env()
			try:
				# Получаем контейнер по ID
				container = client.containers.get(db)
				print('2 container Ok', container)
				print('3 time:', ts + 3600 > time())
				print('3.5 user:', user)

				if container is not None:
					if user is not None and ts + 3600 > time():
						print('4 Ok', "/query.htm#" + db)
						response.headers["X-Token"] = token
						return  ("/query.htm#" + db)
			except:
				print('except')
					# pass



	md5_text = md5((SALT_PSW + data['pswd']).encode('utf-8'));
	response.status_code = checkUser(data['email'], md5_text.hexdigest());

	token = create_access_token({"email": data['email'] , "time": time()})

	response.headers["X-Token"] = token

	return 'Ok'


@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"/tmp/sql/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.get("/start/{token}", response_class=RedirectResponse)
async def start(token: str):
	attrs = None
	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		print('jwt',payload)
		user: str = payload.get("email")
		if user is None:
			raise HTTPException(status_code=401, detail="Invalid authentication credentials")
		ts: int = payload.get("time")

		if time()-ts > 60:		# проверка валидности токена
			raise HTTPException(status_code=419, detail="Timeout Authentification")
	except Exception as e:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

	client = docker.from_env()
	response = Response()
	try:
		container = client.containers.run("selectel",'', 
			volumes={
			'/tmp/sql': { 
				'bind': '/usr/local/pgsql/upload',  # путь в контейнере
				'mode': 'rw'  # режим доступа (чтение/запись)
				}
			},
			detach=True,
			auto_remove=True)
		attrs = container.attrs
		key = attrs['Id'][0:6]

		setkey(key)

	except docker.errors.ImageNotFound as e:
		response.status_code = status.HTTP_404_NOT_FOUND
		return {"error":"image not found"}
	except docker.errors.APIError as e:
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error":"internal"}


	if attrs is None:
		print("Error db is none")

	response = RedirectResponse("/query.htm#" + attrs['Id'][0:6] )
	attrs = container.attrs
	
	response.set_cookie(key="db", value=attrs['Id'][0:6], secure=False)
	token = create_access_token({"email": user , "time": time()})
	response.set_cookie(key="token", value=token , secure=False)
	
	return response



@app.post("/checkdb")
async def setActiveDb(id: Id):
	setkey(id.id)
	return 'Ok'
	
	
@app.put("/sigin")
async def sigin(response: Response, user: User):
	
	try:
		# Establish a connection to the database
		conn = psycopg2.connect(**db_params)
		# Create a cursor object
		cur = conn.cursor()
		code = random.randint(1, 99999999) 

		insert_sql = """
		INSERT INTO users (name, email, password, code)
			VALUES (%s, %s, %s, %s)
		"""
		md5_text = md5((SALT_PSW + user.pswd).encode('utf-8'));

		cur.execute(insert_sql, (user.name, user.email, md5_text.hexdigest(), code))

		# Commit the transaction to save changes to the database
		conn.commit()

	except psycopg2.Error as e:

		print(f"Error inserting data: {e}")
		closeDb(conn, cur)
		response.status_code = status.HTTP_409_CONFLICT
		return  {"error": str(e)}


	closeDb(conn, cur)
	return 'Ok'

@app.get("/parsetoken/{token}")
async def parseToken(token: str):
	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		print('jwt',payload)
		user: str = payload.get("email")
		if user is None:
			raise HTTPException(status_code=401, detail="Invalid authentication credentials")
		ts: int = payload.get("time")

		# if time()-ts > 60:
		# 	raise HTTPException(status_code=419, detail="Timeout Authentification")
	except Exception as e:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

	dt_obj = dt.fromtimestamp(ts)

	return {"user":user, "expire":dt_obj.strftime("%H:%M:%S")}



@app.post("/psql")
async def query(query: Query):

	# проверить токен
	client = docker.from_env()
	try:
		# Получаем контейнер по ID
		container = client.containers.get(query.id)

		print(f"Контейнер {query.id}")
		print(f"Запрос: {query.query}")
		if container is None:
			return f"Контейнер {query.id} не найден"

		ip = container.attrs["NetworkSettings"]["IPAddress"]
		# Выполняем команду внутри контейнера
		print("ip",ip)
		command = '/usr/local/pgsql/bin/psql -c "{}" -h {} -U postgres'.format(query.query, ip)
		print(command)
		# res = os.popen(command).read()
		print('старт процесс')
		res =  subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err_data = res.communicate()
		out = out + err_data

	except docker.errors.NotFound:
		print( f"Контейнер с ID {query.id} не найден")
		return f"Контейнер с ID {query.id} не найден"
	except docker.errors.APIError as e:
		return f"Ошибка Docker API: {str(e)}"
	except Exception as e:
		print (f"Произошла ошибка: {str(e)}", query)
		return f"Произошла ошибка: {str(e)}"
	print('запрос выполнен')
	return PlainTextResponse(out)

@app.post("/token")
async def checkDb(id: Id):
	print('id', id.id)
	client = docker.from_env()
	try:
		container = client.containers.get(id.id)
		if container is None:
			raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
								detail="Db Not Found")
	except docker.errors.NotFound:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
								detail="Db Not Found")
	except docker.errors.APIError as e:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
							detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
							detail=str(e))
	
	return 'Ok'

# кандидат на удаление
@app.get("/run", response_class= RedirectResponse)
async def start():
	client = docker.from_env()
	response = Response()
	try:
		container = client.containers.run("selectel",'', detach=True, auto_remove=True)
		attrs = container.attrs
		key = attrs['Id'][0:6]

		setkey(key)

	except docker.errors.ImageNotFound as e:
		response.status_code = status.HTTP_404_NOT_FOUND
		return {"error":"image not found"}
	except docker.errors.APIError as e:
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error":"internal"}


	response = RedirectResponse("/query.htm#" + attrs['Id'][0:6] )
	attrs = container.attrs
	
	response.set_cookie(key="db", value=attrs['Id'][0:6], secure=False)
	
	return response
# status_code=307


@app.get("/sessions")
async def sessions():

	client = docker.from_env()

	containers = client.containers.list()
	count = len(containers)
	return {"count": count }


@app.post("/cookie")
def create_cookie():
	content = {"message": "Come to the dark side, we have cookies"}
	response = JSONResponse(content=content)
	response.set_cookie(key="fakesession", value="some-session-value")
	return response