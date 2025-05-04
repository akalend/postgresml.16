from typing import Union
from fastapi import FastAPI
from starlette.responses import JSONResponse
from pydantic import BaseModel
import asyncio

from ml.ml import Ml


class Num(BaseModel):
 	num: int


app = FastAPI()


@app.put("/ml", status_code=202)
async def create_item(num: Num):
	ml = Ml()
	ml.run(num.num)
	# await put_item(item)
	return None
