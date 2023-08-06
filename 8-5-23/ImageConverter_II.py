import os
import png
import pydicom as dicom
import argparse
import fnmatch
import gdcm

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--start", type = str, default = "1")
ap.add_argument("-e", "--end", type = str, default = "100")
args = vars(ap.parse_args())

start = int(args["start"])
end = int(args["end"])

def minmax(init_file):
	ds = dicom.read_file(init_file)
	minval = 10000
	maxval = -10000
	for row in ds.pixel_array:
		for col in row:
			if col < minval:
				minval = col
			if col > maxval:
				maxval = col
	
	return (minval, maxval)

def minmaxscan(scan):
	minval = 10000
	maxval = -10000
	files = fnmatch.filter(os.listdir("NGCT{}".format(scan)), '*')
	for i in range(len(files)):
		path = "NGCT{}/{}".format(scan, files[i])
		try:
			(minv, maxv) = minmax(path)
		except:
			minv = minval
			maxv = maxval
		if minv < minval:
			minval = minv
		if maxv > maxval:
			maxval = maxv
	print(minval)
	print(maxval)
	return (minval, maxval)

def converter(scan):
	files = fnmatch.filter(os.listdir("NGCT{}".format(scan)), '*')
	try:
		os.mkdir("SCL/NGCT{}_SCL".format(scan))
	except:
		pass
	step = 0
	for i in range(len(files)):
		try:
			init_file = open("NGCT{}/{}".format(scan, files[i]), "rb")
			(minval, maxval) = minmax(init_file)
			ds = dicom.read_file(init_file)
			shape = ds.pixel_array.shape
			
			diff = maxval - minval
			pixelby = diff / 255.0
				
			image_2d_scaled = []
			for row in ds.pixel_array:
				row_scaled = []
				for col in row:
					col_scaled = int((float(col) - float(minval)) / float(pixelby))
					row_scaled.append(col_scaled)
				image_2d_scaled.append(row_scaled)
			w = png.Writer(shape[1], shape[0], greyscale = True)
			fin_file = open("SCL/NGCT{}_SCL/ngct{}_{}.png".format(scan, scan, step + 1), "wb")
			w.write(fin_file, image_2d_scaled)
			step += 1
		except:
			pass

for i in range(1, 100):
	print(i)
	converter(i)
	print(i)
