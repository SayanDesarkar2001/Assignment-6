# Assignment-6
Predicting Car Prices using ML

Columns and Their Descriptions:
1.	car_ID: Unique identifier for each car.
2.	symboling: Risk factor symbol assigned to the car (-3 to +3).
3.	CarName: Name of the car.
4.	fueltype: Type of fuel used by the car (e.g., gas, diesel).
5.	aspiration: Type of aspiration used in the car (e.g., std, turbo).
6.	doornumber: Number of doors in the car (e.g., two, four).
7.	carbody: Body style of the car (e.g., sedan, hatchback).
8.	drivewheel: Type of drive wheel (e.g., fwd, rwd, 4wd).
9.	enginelocation: Location of the engine (e.g., front, rear).
10.	wheelbase: Wheelbase of the car (in inches).
11.	carlength: Length of the car (in inches).
12.	carwidth: Width of the car (in inches).
13.	carheight: Height of the car (in inches).
14.	curbweight: Curb weight of the car (in pounds).
15.	enginetype: Type of engine (e.g., dohc, ohc, ohcf, ohcv, rotor).
16.	cylindernumber: Number of cylinders in the engine (e.g., two, three, four, five, six, eight, twelve).
17.	enginesize: Size of the engine (in cubic inches).
18.	fuelsystem: Type of fuel system (e.g., mpfi, 2bbl, 1bbl, spdi, spfi, idi).
19.	boreratio: Bore ratio of the engine.
20.	stroke: Stroke of the engine.
21.	compressionratio: Compression ratio of the engine.
22.	horsepower: Horsepower of the car.
23.	peakrpm: Peak RPM of the car.
24.	citympg: City mileage (in miles per gallon).
25.	highwaympg: Highway mileage (in miles per gallon).
26.	price: Price of the car (in dollars).


Manipulation performed on Dataset:

Step 1: Loading and Preprocessing
1.	Load Dataset: Read the CSV file into a DataFrame.
2.	Data Cleaning: Drop missing values.
3.	Define Target and Features: Separate the target variable (price) and convert categorical variables to dummy variables.
4.	Split Data: Split the data into training and testing sets (80-20 split).
5.	Feature Scaling: Standardize the features using StandardScaler.

Step 2: Model Implementation
1.	Define Models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR.
2.	Train and Predict: Train each model and make predictions on the test set.
3.	Evaluate Models: Calculate R-squared, MSE, and MAE for each model. Identify the best model based on R-squared.

Step 3: Feature Importance Analysis
1.	Tree-Based Models: Extract and visualize the top 10 most important features.
2.	Linear Regression: Extract and visualize the top 10 most significant coefficients.

Step 4: Hyperparameter Tuning
1.	Random Forest: Perform Grid Search to find the best hyperparameters and re-evaluate the model.

