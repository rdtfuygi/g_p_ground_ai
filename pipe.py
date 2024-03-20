import win32pipe
import win32file
import struct
import time


class pipe:
	def __init__(self,name:str):
		self.name = '\\\\.\\pipe\\' + name + '_pipe'
		p = 0
		while True:
			try:
				p=win32file.CreateFileW(
					self.name,
					win32file.GENERIC_READ | win32file.GENERIC_WRITE,
					win32file.FILE_SHARE_WRITE,
					None,
					win32file.OPEN_EXISTING,
					0,
					None)
				break
			except:
				time.sleep(1)
		self.pip = p
		return
	
	def __del__(self):
		win32file.CloseHandle(self.pip)
		return
		
	def recept(self) ->list[bool, tuple]:
		_,data_num,_ = win32pipe.PeekNamedPipe(self.pip, 0)
		
		while data_num == 0:
			#time.sleep(1)
			_,data_num,_ = win32pipe.PeekNamedPipe(self.pip, 0)
			
		_,data_buffer = win32file.ReadFile(self.pip,data_num)
		data = struct.unpack('f' * int(len(data_buffer) / 8), data_buffer)
		succeed = False
		if(data[-1] == (len(data) - 1)):
			succeed = True
		return [succeed, data[:-1]]
	
	def send(self,data:list):
		data.append(float(len(data)))
		_,data_num,_ = win32pipe.PeekNamedPipe(self.pip, 0)      
		while data_num != 0:
			#time.sleep(1)
			_,data_num,_ = win32pipe.PeekNamedPipe(self.pip, 0)

		data_buffer = struct.pack('f' * len(data), *data)
		win32file.WriteFile(self.pip,data_buffer)
		return
		
		
		





