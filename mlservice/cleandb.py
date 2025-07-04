#!/usr/bin/env python3

import docker
import sys
import redis

if __name__ == "__main__":
	r = None
	client = docker.from_env()
	containers = client.containers.list(all=True)

	for container in containers:

		if r is None:
			r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

		id = container.id[0:6]
		attrs = container.attrs
		ip = attrs["NetworkSettings"]["IPAddress"]
		# print(f"Container ID: {id} status: {container.status} Ip: {ip}")

		if container.status == 'running':
			res = r.get(id)
			if res is None:
				container = client.containers.get(id)
				print(id, 'deleted')
				container.kill()
			else:
				print(id, 'running')
			# print(f"{id}  Ip: {ip}")


	if r is not None:
		r.quit()