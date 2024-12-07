import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

class FlightPriceModel:
    def __init__(self):
        """Initialize the Flight Price Prediction Models."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        self.X_columns = None
    
    def parse_time(self, time_str):
        """Parse time string to get hour and minute."""
        try:
            time_part = time_str.split()[0]
            hour, minute = map(int, time_part.split(':'))
            return hour, minute
        except Exception as e:
            print(f"Error parsing time {time_str}: {str(e)}")
            return 0, 0
    
    def preprocess_data(self, df):
        """Preprocess the raw flight data."""
        processed_df = df.copy()
        
        # Convert date columns
        processed_df['Journey_Day'] = pd.to_datetime(processed_df['Date_of_Journey'], format='%d/%m/%Y').dt.day
        processed_df['Journey_Month'] = pd.to_datetime(processed_df['Date_of_Journey'], format='%d/%m/%Y').dt.month
        
        # Process departure and arrival times
        dep_times = processed_df['Dep_Time'].apply(self.parse_time)
        arr_times = processed_df['Arrival_Time'].apply(self.parse_time)
        
        processed_df['Dep_Hour'] = dep_times.apply(lambda x: x[0])
        processed_df['Dep_Min'] = dep_times.apply(lambda x: x[1])
        processed_df['Arrival_Hour'] = arr_times.apply(lambda x: x[0])
        processed_df['Arrival_Min'] = arr_times.apply(lambda x: x[1])
        
        # Process duration
        def process_duration(duration):
            if isinstance(duration, str):
                duration_parts = duration.split()
                hours = minutes = 0
                for part in duration_parts:
                    if 'h' in part:
                        hours = int(part.replace('h', ''))
                    if 'm' in part:
                        minutes = int(part.replace('m', ''))
                return hours, minutes
            return 0, 0
        
        durations = processed_df['Duration'].apply(process_duration)
        processed_df['Duration_Hours'] = durations.apply(lambda x: x[0])
        processed_df['Duration_Mins'] = durations.apply(lambda x: x[1])
        
        # Handle stops
        processed_df['Total_Stops'] = processed_df['Total_Stops'].replace({
            'non-stop': 0, '1 stop': 1, '2 stops': 2, 
            '3 stops': 3, '4 stops': 4
        })
        
        # Create dummy variables
        categorical_columns = ['Airline', 'Source', 'Destination']
        processed_df = pd.get_dummies(processed_df, columns=categorical_columns)
        
        # Drop unnecessary columns
        columns_to_drop = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 
                          'Duration', 'Route', 'Additional_Info']
        processed_df.drop(columns_to_drop, axis=1, inplace=True)
        
        return processed_df

    def train(self, data_path, models_dir='models'):
        """Train multiple models and save them separately."""
        print("Loading data...")
        df = pd.read_excel(data_path)
        
        print("Preprocessing data...")
        processed_df = self.preprocess_data(df)
        
        # Save column names for prediction
        self.X_columns = processed_df.drop('Price', axis=1).columns.tolist()
        
        # Split features and target
        X = processed_df.drop('Price', axis=1)
        y = processed_df['Price']
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Train and save each model
        results = {}
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'Train R² Score': train_score,
                'Test R² Score': test_score,
                'MAE': mae
            }
            
            # Save model
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            model_data = {
                'model': model,
                'columns': self.X_columns,
                'metrics': {
                    'train_score': train_score,
                    'test_score': test_score,
                    'mae': mae
                }
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Saved {model_name} to {model_path}")
        
        # Print comparison of models
        print("\nModel Comparison:")
        comparison_df = pd.DataFrame(results).round(3)
        print(comparison_df)
        
        # Save model comparison
        comparison_path = os.path.join(models_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path)
        print(f"\nSaved model comparison to {comparison_path}")

if __name__ == "__main__":
    model = FlightPriceModel()
    model.train("Data_Train.xlsx")