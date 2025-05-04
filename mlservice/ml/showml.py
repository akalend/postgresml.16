import os,sys,time,psutil

"""
Return status of process
"""
def show():

	found = False

	print("pid\t   args\t   status")
	print("---------------------------")
	for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
		if proc.name() == 'python' and proc.cmdline()[1][-6:] == 'mlexec':
			# print(proc.info)
			if proc.info['pid'] == os.getpid():
				continue				
			found = True
			print("%s\t| %s\t| %s"%(proc.info['pid'], proc.info['cmdline'][2], proc.status()))
			# if not found:
			# 	print("no process")
		
	if not found:
		print("Process ml not found")
