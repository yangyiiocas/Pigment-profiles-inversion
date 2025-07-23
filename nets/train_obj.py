import torch
import torch.nn as nn
import numpy as np
import os,shutil,datetime,pickle,random

import nets.train as train

class training_object():
	def __init__(self, param,device,learning_rate,epochs,cv_num):
		self.param = param
		self.device = device
		self.learning_rate =learning_rate
		self.epochs = epochs
		self.cv_num = cv_num

		self.x1 = None
		self.x2 = None
		self.y1 = None
		self.y2p = None
		self.y2d = None

		self.y2cal = None
		self.test_x1 = None
		self.test_x2 = None
		self.test_y1 = None
		self.test_y2p = None
		self.test_y2d = None
		self.test_y2cal = None
		self.test_y2cal = None

		
		

		self.LOSS_train = []
		self.LOSS_val = []
		self.LOSS_test = []

		self.predict_cv = None
		self.predict_test = None




