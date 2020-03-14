There are four folders: Fully, cnn, locally, and ensemble.

When running the fully connected neutron networl, cd fully folder. 

	for task1, running "handwrite_fully.py", and then you will get the "Fully.png" file.

	for task2, running "fully_parameter.py". 
	
	1. If you want to run the different initializers, you need to delete the "#" in front of "for i in range(0,13). (66th line) and keep all other #. I run different three by three, you need to adjust the number. if not, you will get the 13 different curve.

	about the 69th line, it is for getting the better three initializers. It is only for compare.
	after running initializers, you need to put # in 197th and 198th lines.	

	2. about comparing different learning rate, you need delete the "#" in front of "for j in range(4)"(73rd line) until 90th line. And keep the others "#" including the question 1 # above.

	3. about comparing different batch size, you need to delete the # in 94th-127th lines, and keep others.

	4. about comparing different momentum, you need to delete the # in 130th-148th lines, and keep others.

	Every question, you need to modify the the name in plt.savefig('').


	for task3, running fully_dropout.py and fully_l1.py.



CNN:
	for task1, running "handwrite_cnn.py".
	for task2, 
		different initializers: delete # in 95th line. keep others.
	
	after running initializers, you need to put # in 180th and 181st lines.
	
		different learning rate: 100-117 lines.
		
		different batch size: 122-155

		different momentum: 158-175

	for task3, 
		different dropout: running "cnn_dropout.py"

		different l1: running "cnn_l1.py"



Locally:
	for task1, running "handwrite_locally.py".

	for task2, running "locally_parameter.py".

		different initializers: 64th line.

		after initializers: you need put "#" in 179 and 180th lines in order to get good curve.
		
		different learning rate: 69th-86th lines.
		
		different batch size: 91st-124th lines.

		different momentum: 127th-144th lines.

	for task3, running locally_dropout.py and locally_l1.py.



For ensemble, cd ensemble_all folder.
	running ensemble_getmodel.py to get model of six.
	running ensemble_getaccu.py to get result.
