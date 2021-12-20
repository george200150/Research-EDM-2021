import numpy as np


def read_students(path, prediction = "regression", expected_number_of_features = 4):
	"""
	Returneaza datele studentilor citite din fisierul de la path in 2 variabile (ndarrays):
		grades e o matrice, cu un student pe fiecare rand
		final_grade este un vector cu nota finala a fiecarui student
	prediction: seteaza modul de salvare a notei finale:
		"regression" - se retine nota finala
		"classification" - se retine 0 daca a picat si 1 daca a trecut
	expected_number_of_features: cate note ar trebui sa fie FARA nota finala
		E nevoie pentru a verifica daca sunt date lipsa
	"""
	file = open(path)
	grades = []
	passed = []
	line_no = 0
	for line in file:
		line_no += 1
		grades_str = line.strip().split()
		student_grades = [float(g) for g in grades_str[:-1]]

		if len(student_grades) != expected_number_of_features:
			print("line: " + str(line_no) + ", features: " + str(student_grades))
		else:
			grades.append(student_grades)
			if prediction == "classification":
				if float(grades_str[-1]) >= 5.0:
					passed.append(1)
				else:
					passed.append(0)
			elif prediction == "regression":
				passed.append(float(grades_str[-1]))
			else:
				raise ValueError("'prediction' parameter can only have the following values: 'classification' or 'regression'")
	grades = np.array(grades)
	final_grade = np.array(passed) 
	return grades, final_grade


path = r"C:\Users\Andrei Mihai\Desktop\andrei workplace\Python\doctorat\src\EDM\data"

if __name__ == "__main__":
	# Import training dataset
	train_x, train_y = read_students(path + '\\train3.txt')
	
	# Import testing dataset
	test_x, test_y = read_students(path + '\\test3.txt')
	
	print(np.shape(train_x))
	print(np.shape(train_y))
	print(np.shape(test_x))
	print(np.shape(test_y))