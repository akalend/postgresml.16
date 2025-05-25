from typing import Union
from fastapi import FastAPI,Request
from starlette.responses import JSONResponse
from pydantic import BaseModel
import asyncio

from ml.ml import Ml


class Indata(BaseModel):
	num: int
	ip: str


app = FastAPI()

@app.get("/echo")
async def echo(request: Request):
	ip_address = request.client.host
	return {"ip_address": ip_address}

@app.put("/ml", status_code=202)
async def create_model(indata: Indata):
	ml = Ml()
	print(indata)
	ml.run(indata.num)
	return None
