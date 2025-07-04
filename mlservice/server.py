from typing import Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi import HTTPException, Depends, status, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer

from starlette.responses import JSONResponse
from pydantic import BaseModel
from hashlib  import md5
import asyncio, os
from time import sleep, time
import datetime, subprocess
import redis
import docker
import jwt
import psycopg2
import random

from passlib.context import CryptContext

# Secret key for signing JWT tokens. We'll provide example including secret key rotation below.
SECRET_KEY = "AZ03;"

# Expire session time in sec
EXPIRE_SESSION = 100

# Algorithm used for JWT token encoding
ALGORITHM = "HS256"
SALT_PSW="23.xZ%"

db_params = {
	"dbname": "users",
	"user": "postgres",
	"password": None,
	"host": "127.0.0.1",
	"port": "5432"
}

app = FastAPI()

class Indata(BaseModel):
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

			if time()-ts > 600:
				raise HTTPException(status_code=419, 
					detail="Timeout Authentification")
		except Exception as e:
			print(e)
			raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
								detail="Invalid authentication credentials 3")
		return True


def checkUser(email, pswd):
	try:
		conn = psycopg2.connect(**db_params)
		cur = conn.cursor()

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
async def create_model(indata: Indata):
	print('new request',indata)
	r = redis.Redis(host='127.0.0.1')
	out = "{}:{}".format( indata.ip, indata.num) 
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
async def sigout(response: Response, user: User):

	md5_text = md5((SALT_PSW + user.pswd).encode('utf-8'));
	response.status_code = checkUser(user.email, md5_text.hexdigest());

	token = create_access_token({"email": user.email , "time": time()})
	response.headers["X-Token"] = token

	return 'Ok'

@app.get("/start/{token}", response_class=RedirectResponse)
async def start(token: str):
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

	response = RedirectResponse(url="/get.htm")
	response.set_cookie(key="token", value=token)
	response.status_code = status.HTTP_302_FOUND
	return response


@app.post("/checkdb")
async def checkdb(id: Id):
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
# 	print('new request',user)
	return 'Ok'


@app.post("/psql")
async def query(query: Query):

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
	
	response.set_cookie(key="id", value=attrs['Id'][0:6], max_age=3600, secure=False, httponly=True)
	
	return response
# status_code=307



@app.get("/sessions")
async def sessions( authorized: bool = Depends(verify_token)):

	print('authorized', authorized)
	# if not authorized:
	# 	return {"count":10 }
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