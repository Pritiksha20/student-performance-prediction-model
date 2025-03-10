import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv('student_exam.csv')

# Feature Selection
X = data[['StudyHours', 'PreviousScores']]
y = data['FinalScore']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Test with custom input
def predict_performance(study_hours, previous_scores):
    prediction = model.predict([[study_hours, previous_scores]])
    return (prediction[0])

# Test Input
study_hours = float(input("Enter Study Hours per day: "))
previous_scores = float(input("Enter Previous Scores: "))
sol = int(predict_performance(study_hours, previous_scores))
if sol==1:
    print("final score is pass")
else:
    print("final score is fail")
