import numpy as np
import quandl as qd
import matplotlib.pyplot as plt 
import pandas as pd
import sys

#Give the first argument as the input file
#Second argument as the X and third as Y feature to be compared with

def sse(m,b,datas):
	total_error=0
	#Using the formula Sum_Of( |(Y - F(X))**2| ) / N where N is number of data points
	#The only intution that this gives us (NON STATISTICAK VIEW POINT) is how accurate the model is
	#extremely smaller value may be an issue since the model might get Over fitted
	#extremely big and the model is ineeficient 
	for data in datas:
		X=data[0]
		Y=data[1]
		fx=X*m + b
		total_error+=(Y-fx)**2
	return total_error / float(len(datas))


def gradient_based_values(m_current , b_current , datas , rate):
	m_gradient=0
	b_gradient=0
	N=float(len(datas))
	for data in datas:
		x=data[0]
		y=data[1]
		#Differentiate the SSE function.
		#That is the error function that we need to minimize. The squaring of the number in F(e)
		#is just for having postive values at each point
		#minimiing this function is what will allow us for a properly regressed linar line
		#which each iteration of x and y we get new values of gradient of m and b
		#these values are then multiplied with a decreasing factor
		#that is learning rate and then substracted of from the current value
		b_gradient+=(-2 / N )*(y - (m_current*x + b_current))
		m_gradient+=(-2 / N )*(y - (m_current*x + b_current))*x
	m_new=m_current - m_gradient*rate
	b_new=b_current - b_gradient*rate
	return (m_new, b_new)	




def gradient_descent_demo(dataset_file,X,Y,rate,epoch):
	print('Gradient Descent on any data set of any two columns : \n')
	dataset = pd.read_csv(dataset_file)
	dataset = dataset[pd.notnull(dataset[X])]
	dataset = dataset[pd.notnull(dataset[Y])]
	datas=dataset.as_matrix(columns=[X , Y])
	datas_for_plot= datas.copy()
	datas_for_plot=datas_for_plot.transpose()
	
	#for the eqution Y = mX + b
	m_value = 0 #assumed m
	b_value = 0 #assumed b
	
	print('Assumed m is',m_value ,
		  ' and assumed b is',b_value,
		  'net error from sum of square error method :',sse(m_value , b_value, datas))
	
	maximum=np.max(datas_for_plot[0])
	X=np.linspace(0,maximum,2)
	
	plt.ion()
	for i in range(epoch):
		#each iteration for changing m and b values repeatedly untill it optimises
		m_value,b_value = gradient_based_values(m_value , b_value , datas , rate)
		print(m_value,b_value,sse(m_value , b_value , datas))
		
		#X and Y relation
		Y=X*m_value + b_value

		#plots for the values and the line
		plt.clf()
		plt.scatter(datas_for_plot[0],datas_for_plot[1],color='blue')
		plt.plot(X,Y,color='yellow')
		plt.pause(0.0001)
	plt.show()	


def check_CommandLine_arguements():
	if(len(sys.argv) < 6):
		print('YOU MISSED ARGUMENTS :gradient_descent.py file_name X Y')
		exit()
		
	

if __name__ == '__main__':
	check_CommandLine_arguements()
	file_name,Xfeature,Yfeature,rate,epoch=sys.argv[1:]
	rate=float(rate)
	epoch=int(epoch)
	gradient_descent_demo(file_name,Xfeature,Yfeature,rate,epoch)
