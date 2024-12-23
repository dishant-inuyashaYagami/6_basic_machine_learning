import csv
import os
import numpy as np
from matplotlib import pyplot as plt
import time

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


# ==================================== Computes Gradient at Parameters (a,b) =============================================================================
def comp_grad(a, b, grad_val, error):
	if num_iter % 1000 == 0:
		print ("computing gradient at all points")
	avg_grad = [0, 0]
	for index in range(0, len(acidity)):					# taking all input data for computing gradient
		x  = acidity[index]
		y  = density[index]
		ht = a*x + b
		pt_error = y - ht														# error at a point
		grad_val[0]   += pt_error*x
		grad_val[1]   += pt_error
		error[0]  += pt_error*pt_error
	avg_grad[0]    =  grad_val[0]/len(acidity)
	avg_grad[1]    =  grad_val[1]/len(acidity)
	grad_val	   =  avg_grad
	if num_iter % 1000 == 0:
		print("total error: " + str(error[0]));


# ==================================== MAIN GD CALL ==================================================================================
learning_rate = 0.0001
error_difference = 1      # difference between subsequent error values
old_error    = [1]
flag = 0
direction = 1
num_iter = 0
start = time.time()
while error_difference > 0.000000001 and num_iter < 100000:
	grad_val  = [0,0]
	error = [0.0]
	comp_grad (slope, intercept, grad_val, error)								# computing gradient
	slope     = slope     - direction*learning_rate*grad_val[0];		# updating hypothesis
	intercept = intercept - direction*learning_rate*grad_val[1];		# updating hypothesis
	

	if flag == 0:						# change direction if the error is not decreased after the first step
		new_grad = [0,0]
		new_error = [0]
		comp_grad(slope, intercept, new_grad, new_error)
		if new_error[0] > error[0]:
			print ("changing the direction of movement! restoring previous hypothesis . . .") 
			slope 	  += direction*learning_rate*grad_val[0];
			intercept += direction*learning_rate*grad_val[1];
			direction = -1
			num_iter += 1
			continue
		else:
			print ("moving in the right direction... all good!") 
		flag = 1
	error_difference   =  abs(old_error[0] - error[0]);
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
name = "gradient_descent.png"
plt.savefig(name, bbox_inches='tight')








