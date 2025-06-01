from typing import Union
from fastapi import FastAPI,Request, BackgroundTasks
from starlette.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from time import sleep
import datetime
import docker
from ml.ml import Ml


class Indata(BaseModel):
	num: int
	ip: str



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

@app.get("/sleep/{id}")
async def send_notification(id: int, background_tasks: BackgroundTasks):
	background_tasks.add_task(task_sleep, id)
	return None


@app.post("/start")
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
	return {"start":"Ok"}


@app.get("/sessions")
async def sessions():
	client = docker.from_env()

	containers = client.containers.list()
	count = len(containers)
	return {"count": count }
