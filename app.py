import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    global model
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Read the CSV file
            data = pd.read_csv(filepath)
            
            # Assuming the last column is the target variable and the rest are features
            X = data.iloc[:, :-1]  # Features
            y = data.iloc[:, -1]   # Target variable
            
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Save the model to a file
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)

            return redirect(url_for('home'))

    return render_template('index.html', model_exists=model is not None)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return "Model not trained. Please upload a CSV file first."
    
    # Get user input from the form
    user_input = request.form.get('input_value')
    try:
        # Convert input to the appropriate type (float)
        input_value = float(user_input)
        
        # Predict the output
        prediction = model.predict([[input_value]])
        return f'Predicted Output: {prediction[0]}'
    except ValueError:
        return "Invalid input. Please enter a numeric value."

if __name__ == '__main__':
    app.run(debug=True)
