from typing import Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, PlainTextResponse
from starlette.responses import JSONResponse
from pydantic import BaseModel
import asyncio, os
from time import sleep
import datetime, subprocess
import docker
from ml.ml import Ml


class Indata(BaseModel):
	num: int
	ip: str

class Query(BaseModel):
	query: str
	id: str


def task_sleep(tim):
	with open('log.txt', 'a') as f:
		now = datetime.datetime.now()
		f.write( "start {}\n".format(now))
		sleep(tim)
		now = datetime.datetime.now()
		f.write( "stop {}\n".format(now))


app = FastAPI()

@app.get("/echo")
async def echo(request: Request):
	ip_address = request.client.host
	return {"ip_address": ip_address}

@app.put("/ml", status_code=202)
async def create_model(indata: Indata):
	
	ml = Ml()
	print(indata)
	ml.run(indata.num, indata.ip)
	return None

@app.get("/")
async def home():
	return '{"result":"Ok"}'

# @app.get("/sleep/{id}")
# async def send_notification(id: int, background_tasks: BackgroundTasks):
# 	background_tasks.add_task(task_sleep, id)
# 	return None

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
		res =  subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()

	
	except docker.errors.NotFound:
		print( f"Контейнер с ID {query.id} не найден")
		return f"Контейнер с ID {query.id} не найден"
	except docker.errors.APIError as e:
		return f"Ошибка Docker API: {str(e)}"
	except Exception as e:
		print (f"Произошла ошибка: {str(e)}", query)
		return f"Произошла ошибка: {str(e)}"

	return PlainTextResponse(res)

@app.get("/run")
async def start():
	client = docker.from_env()
	try:
		container = client.containers.run("selectel:latest",'', detach=True, auto_remove=True)
	except docker.errors.ImageNotFound as e:
		response.status_code = status.HTTP_404_NOT_FOUND
		return {"error":"image not found"}
	except docker.errors.APIError as e:
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error":"internal"}

	attrs = container.attrs
	# # response.set_cookie(key="id", value=attrs['Id'][0:6], max_age=300, secure=False, httponly=True)

	return RedirectResponse(url="/query.htm#" + attrs['Id'][0:6] ) 
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