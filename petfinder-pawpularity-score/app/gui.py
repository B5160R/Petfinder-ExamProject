import tkinter as tk
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_curve
from sklearn.model_selection import cross_val_score

class Application(tk.Tk):
  
	def __init__(self):
		super().__init__()
		self.title("Petfinder Pawpularity Score")
		self.geometry("550x750")
		self.create_widgets()

	def create_widgets(self):
		# Create title label and set font size and weight
		title_label = tk.Label(self, text="Petfinder Pawpularity Score Predictor", font=("Helvetica", 16, "bold"))
		title_label.grid(row=0, column=0, columnspan=6, padx=5, pady=5)

		# Checkbox labels and variables
		checkbox_labels = [
			"Subject Focus", "Eyes", "Face", "Near", "Action",
			"Accessory", "Group", "Collage", "Human", "Occlusion",
			"Info", "Blur"
		]
		self.checkbox_vars = []
  
		# Create checkboxes
		for i, label in enumerate(checkbox_labels):
			var = tk.BooleanVar()
			checkbox = tk.Checkbutton(self, text=label, variable=var)
			checkbox.grid(row=i//6 + 1, column=i%6, padx=5, pady=5)
			self.checkbox_vars.append(var)
		
		# Create submit button
		submit_button = tk.Button(self, text="Submit", command=self.run_model)
		submit_button.grid(row=len(checkbox_labels)//6+1, column=0, columnspan=6, padx=5, pady=5)

	  # Create metrics button
		metrics_button = tk.Button(self, text="Show Performance Metrics", command=self.performanceMeasure)
		metrics_button.grid(row=len(checkbox_labels)//6+1, column=3, columnspan=6, padx=5, pady=5)

		# Create result label
		self.result_label = tk.Label(self, text="")
		self.result_label.grid(row=len(checkbox_labels)//6+2, column=0, columnspan=6, padx=5, pady=5)
  
		# Create a lable to display the ids of the pets
		self.pet_id_label = tk.Label(self, text="")
		self.pet_id_label.grid(row=len(checkbox_labels)//6+3, column=0, columnspan=6, padx=5, pady=5)
  
		# Create a label to display the actual score of the pet
		self.pet_score_label = tk.Label(self, text="")
		self.pet_score_label.grid(row=len(checkbox_labels)//6+4, column=0, columnspan=6, padx=5, pady=5)
  
		# Display the image of the pet
		self.pet_image_label = tk.Label(self)
		self.pet_image_label.grid(row=len(checkbox_labels)//6+5, column=0, columnspan=6, padx=5, pady=5)
  
	def run_model(self):
		# Get the selected checkbox values
		selected_values = [var.get() for var in self.checkbox_vars]

		# Load the regression model
		model = joblib.load('../model/regression_model.pkl')

		# Prepare the input data for the model
		input_data = [int(value) for value in selected_values]

		print(input_data)

		# Run the model prediction
		prediction = model.predict([input_data])

		# Display the prediction result
		self.result_label.config(text=f"Predicted Pawpularity Score: {prediction[0]:.2f}")

		# Find the id of the pet based on the input data and Pawpularity Prdiction Score
		pet_id_and_score = self.get_pet_id_and_score(prediction[0], input_data)
  
		# Display the pet id
		self.pet_id_label.config(text=f"Pet Id: {pet_id_and_score[0]}")
  
		# Display the actual score of the pet
		self.pet_score_label.config(text=f"Actual Pawpularity Score: {pet_id_and_score[1]}")

		# Display image of the pet with id from the data folder
		pet_image_path = f"../data/train/{pet_id_and_score[0]}.jpg"
		pet_image = Image.open(pet_image_path)

		# Resize the image
		pet_image = pet_image.resize((300, int(300 * pet_image.size[1] / pet_image.size[0])), Image.ADAPTIVE)
		pet_image = ImageTk.PhotoImage(pet_image)

		# Display the image
		self.pet_image_label.config(image=pet_image)
		self.pet_image_label.image = pet_image
  
	def get_pet_id_and_score(self, score, input_data):
		# Load the train.csv file
		df = pd.read_csv('../data/train.csv')
		
		# Find the pet indexes that match the input data
		pet_indexes = df[
			(df['Subject Focus'] == input_data[0]) &
			(df['Eyes'] == input_data[1]) &
			(df['Face'] == input_data[2]) &
			(df['Near'] == input_data[3]) &
			(df['Action'] == input_data[4]) &
			(df['Accessory'] == input_data[5]) &
			(df['Group'] == input_data[6]) &
			(df['Collage'] == input_data[7]) &
			(df['Human'] == input_data[8]) &
			(df['Occlusion'] == input_data[9]) &
			(df['Info'] == input_data[10]) &
			(df['Blur'] == input_data[11])
		].index.tolist()
		
		# Find the pet index that has the closest Pawpularity score to the predicted score
		pet_index = pet_indexes[0]
		min_diff = abs(score - df.loc[pet_index, 'Pawpularity'])
		for index in pet_indexes:
			diff = abs(score - df.loc[index, 'Pawpularity'])
			if diff < min_diff:
				min_diff = diff
				pet_index = index
		
		# Find and return the pet id and the actual score
		pet_id_and_score = df.loc[pet_index, ['Id', 'Pawpularity']]
		return pet_id_and_score

	def performanceMeasure(self):
		# Load the test data
		test_data = pd.read_csv('../data/train.csv')

		# Load the regression model
		model = joblib.load('..model/regression_model.pkl')

		# Prepare the input data for the model
		X = test_data.drop(columns=['Id', 'Pawpularity'])
		y_true = test_data['Pawpularity']

		# Run the model prediction
		y_pred = model.predict(X).round()
  
		# Calculate performance measures
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, average='weighted')
		recall = recall_score(y_true, y_pred, average='weighted')

    # Perform cross-validation
		cv_scores = cross_val_score(model, X, y_true, cv=3)
  
		# show the performance measures
		self.result_label.config(text=f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nCV Scores: {', '.join(map(str, cv_scores))}")

if __name__ == "__main__":
	app = Application()
	app.mainloop()