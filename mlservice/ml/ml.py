#!/usr/bin/env python

import sys, os, time
import subprocess
import psutil
from signal import SIGTERM

import sys,time

class Ml():
	def __init__(self):
		pass

	def run(self, number, ip):
		pwd = os.getenv("PWD")
		cmd = '{}/ml/mlexec {} {}'.format(pwd, number, ip)
		print(cmd)
		subprocess.run('{}/ml/mlexec {}'.format(pwd, number), shell=True, cwd=pwd)


if __name__ == "__main__":
	ml = Ml()
	n = '12345'
	ml.run(n)