import numpy as np 
import datetime
import os
import pickle

class Dataset():
	def __init__(self, date, lon, lat, depth):
		self.datetime = date
		self.lon = lon
		self.lat = lat
		self.depth = depth

		self.source = None
		self.pigments = None
		self.pigments_use = None
		self.satellite_match = None
		self.satellite_match_use = None

		self.data_use = None

		self.index = []

	def add_index(self,a):
		self.index.append(a)

	def add_source(self, a):
		self.source = a
	
	def add_pigments(self, a):
		self.pigments = a

	def add_pigments_use(self, a):
		self.pigments_use = a 

	def add_satellite_match(self, a):
		self.satellite_match = a

	def add_satellite_match_use(self, a):
		self.satellite_match_use = a

	def revise_use(self,):
		self.data_use = {var:(self.pigments_use[var]&self.satellite_match_use) for var in self.pigments}



