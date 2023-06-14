import os
import png
import pydicom as dicom
import argparse
import fnmatch
import gdcm

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--ptn", type = str)
ap.add_argument("-m", "--maxi", type = str, default = "0")
ap.add_argument("-t", "--type", type = str, default = "NG")
args = vars(ap.parse_args())

ptn = args["ptn"]
maxi = int(args["maxi"])
dtype = args["type"].upper()


def converter(init_file, fin_file, maxi):

	ds = dicom.read_file(init_file)
	shape = ds.pixel_array.shape
	
	image_2d = []
	max_val = 0
	
	if maxi == 0:
		for row in ds.pixel_array:
			pixels = []
			for col in row:
				pixels.append(col)
				if col > max_val:
					max_val = col
			image_2d.append(pixels)
	else:
		for row in ds.pixel_array:
			pixels = []
			for col in row:
				pixels.append(col)
			image_2d.append(pixels)
		
		max_val = maxi
	
	image_2d_scaled = []
	for row in image_2d:
		row_scaled = []
		for col in row:
			col_scaled = int((float(col) / float(max_val)) * 255.0)
			if col_scaled > 255:
				col_scaled = 255
			if col_scaled < 0:
				col_scaled = 0
			row_scaled.append(col_scaled)
		image_2d_scaled.append(row_scaled)
	
	w = png.Writer(shape[1], shape[0], greyscale = True)
	w.write(fin_file, image_2d_scaled)

def executor(init_file_path, fin_file_path, maxi):
	
	if not os.path.exists(init_file_path):
		raise Exception('File {} does not exist'.format(init_file_path))
	
	init_file = open(init_file_path, 'rb')
	fin_file = open(fin_file_path, 'wb')
	
	converter(init_file, fin_file, maxi)
	
	fin_file.close()	

for i in range(int(ptn), int(ptn) + 1):
	files = fnmatch.filter(os.listdir("{}CT{}".format(dtype, i)), '*')
	files.sort()
	num_files = len(files)
	pre = int(files[0][:4])
	start = int(files[0][4:])
	print(i)
	for place in range(num_files): # put num_files here
		suffix = (place * 11) + start
		strsuffix = str(suffix)
		if len(strsuffix) < 4:
			strsuffix = "0" + strsuffix
		intsuffix = int(strsuffix)
		num = intsuffix + (10000 * int(pre))
		name = "{}CT{}/".format(dtype, i) + str(num)
		path = '{}CT{}_IMG/{}ct{}_{}.png'.format(dtype, i, dtype.lower(), i, place + 1)
		try:
			executor(name, path, maxi)
		except:
			pass
	print(i)
