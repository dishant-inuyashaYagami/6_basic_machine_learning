import csv
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import sys

print ("Arguments: " + str(sys.argv))
print ("Number of arguments: " + str(len(sys.argv)))
if len(sys.argv) != 2:
	print ("Please give exactly one command line argument: <batch_size>")
	exit(0)
if int(sys.argv[1]) < 1 or int(sys.argv[1]) > 100:
	print ("Invalid Argument! Please give the argument in range [1,100] since the num of data points in the set are 100.")
	exit(0) 
batch_size = int(sys.argv[1]);



print ("\n=========== Regular/Stochastic/Batch GRADIENT DECENT ==================")

# ==================================== Reading Data =====================================================================
acidity = []
density = []
__location__ = os.path.realpath(os.path.join(os.getcwd(), "linearX.csv"))
print ("fetching data from location: " + __location__ + " . . .")
with open(__location__, 'r') as csvfile:   # Open the CSV file in read mode # Create a reader object
  csv_reader = csv.reader(csvfile)
  for row in csv_reader:  # Iterate through the rows in the CSV file
    acidity.append(float(row[0]))     # row is a list containing single element

__location__ = os.path.realpath(os.path.join(os.getcwd(), "linearY.csv"))
print ("fetching data from location: " + __location__ + " . . .")
with open(__location__, 'r') as csvfile:   # Open the CSV file in read mode # Create a reader object
  csv_reader = csv.reader(csvfile)
  for row in csv_reader:  # Iterate through the rows in the CSV file
  	density.append(float(row[0]))     # row is a list containing single element

print ("number of data points: " + str(len(acidity)))
if len(density) != len(acidity):
	print ("ERROR! Input number of attributed does not match the number of class labels!")
	exit(0)


# ==================================== Plot Data =====================================================================
plt.scatter(acidity,density)


# ==================================== Initialization ======================================================================================
# starting point: line joining any two points, slope = 1; y = ax + b;     // can be varied as experiment continues to get better results
# learing rate = 0.001										  // can be varied as experiment continues to get better results
# stopping criteria; when error <= 0.001					  // can be varied as experiment continues to get better results

# based on first two points
slope     = (density[1] - density[0])/(acidity[1] - acidity[0])
intercept =  density[0] - acidity[0]*slope
print("initial slope: " + str(slope) + ", initial intercept: " + str(intercept))
plt.axline((acidity[0], density[0]), slope=slope, linewidth=1, color='b')			# or plt.plot([acidity[0],acidity[1]], [density[0],density[1]])


# ==================================== Computes Batch Gradient at Parameters (a,b) =============================================================================
def comp_grad(a, b, grad_val, start_range, end_range):
	if num_iter % 1000 == 0:
		print ("computing gradient at all points")
	avg_grad = [0, 0]
	for index in range(start_range, end_range):					# takes only a subset of points for computing gradient
		x  = acidity[index]
		y  = density[index]
		ht = a*x + b
		pt_error = y - ht
		grad_val[0]   += pt_error*x
		grad_val[1]   += pt_error
	avg_grad[0]    =  grad_val[0]/(end_range - start_range)
	avg_grad[1]    =  grad_val[1]/(end_range - start_range)
	grad_val	   =  avg_grad

	if num_iter % 1000 == 0:
		print("average error: " + str(error[0]));

def comp_error(a, b, error):
	for index in range(0, len(acidity)):					# compute the full average error
		x  = acidity[index]
		y  = density[index]
		ht = a*x + b
		pt_error = y - ht
		error[0]  += pt_error*pt_error


# ==================================== MAIN GD CALL ==================================================================================
learning_rate = 0.0001
error_difference = 1      # difference between subsequent error values
old_error    = [1]
flag = 0
direction = 1
num_iter = 0

# error_difference > 0.000000001
start = time.time()
while (old_error[0] > 0.0000001) and num_iter < 100000:
	
	error = [0]
	comp_error(slope, intercept, error)

	grad_val  = [0,0]
	batch_start_range = (num_iter*batch_size)%len(acidity)
	if batch_size == len(acidity):
			batch_end_range = len(acidity)
	else:
			batch_end_range   = ((num_iter*batch_size + batch_size)%len(acidity))
	comp_grad (slope, intercept, grad_val, batch_start_range, batch_end_range)								# computing gradient
	slope     = slope     - direction*learning_rate*grad_val[0];		# updating hypothesis
	intercept = intercept - direction*learning_rate*grad_val[1];		# updating hypothesis

	if flag == 0:						# change direction if the error is not decreased after the first step
		new_error = [0]
		comp_error(slope, intercept, new_error)
		if new_error[0] > error[0]:
			print ("changing the direction of movement! restoring previous hypothesis . . .") 
			slope 	  += direction*learning_rate*grad_val[0];
			intercept += direction*learning_rate*grad_val[1];
			direction = -1
			continue
		else:
			print ("moving in the right direction... all good!") 
			flag = 1	

		# sometimes stuck here since gradient is computed over a batch; it might not always decrease the overall average error
		# this is required as well since for small batches the error diverges after some time
	# error_difference   =  old_error[0] - error[0];			
	# if error_difference < 0:
	# 	print ("previous error: " + str(old_error[0]))
	# 	print ("new      error: " + str(error[0]))
	# 	print ("EXITING . . . stuck in local minium!")

	old_error[0]   =  error[0]
	num_iter +=1;

print ("number of iterations done: " + str(num_iter))
end = time.time()

plt.axline((0, intercept), slope=slope, linewidth=1, color='r')			# or plt.plot([acidity[0],acidity[1]], [density[0],density[1]])
#plt.show()
error_display = "error value:   " + str(old_error[0]) + "\n"
time_display  = "time consumed: " + str(int(end - start)) + " sec."
plt.text(0, 1, error_display + time_display, fontsize = 10, 								# print error value
         bbox = dict(facecolor = 'red', alpha = 0.5))
name = "batch_size_" + str(batch_size) + ".png"
plt.savefig(name, bbox_inches='tight')







