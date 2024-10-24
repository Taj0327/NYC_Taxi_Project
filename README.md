# NYC Taxi Project

This project analyzes NYC taxi data to train machine learning models for predictions or data analysis. The project includes code for data loading, cleaning, preprocessing, as well as model training and evaluation. For the modelling and analysis part, the project uses a multi-layer perceptron (MLP) for prediction.

## Dependencies

To run the project, you'll need the following dependencies. These can be installed using `pip`:

```{bash}
pip install -r requirements.txt
```

## Project Sturcture

NYC_Taxi_Project/
- data/                       # Folder for datasets
  - yellow_tripdata_2024-01.parquet
- models/                     # Folder for saving trained models
- notebooks/                  # Jupyter notebooks for exploration (optional)
- scripts/                    # Utility scripts for data and models
  - data_utils.py             # Functions for data loading, cleaning, and splitting
  - model_utils.py            # Dataset class, model architecture, and training code
- train_model.py              # Main script to load data, process, and train models
- README.md                   # Project documentation (this file)
- requirements.txt            # Python dependencies


## How to run the code

### Step 1: Clone the Repository

First, clone the project repository from GitHub (replace with your actual repository URL):

```{bash}
git clone https://github.com/your-username/NYC_Taxi_Project.git
cd NYC_Taxi_Project
```

### Step 2: Set Up the Python Environment

(Optional but recommended) Create a virtual environment and activate it.

```{bash}
python3 -m venv venv
source venv/bin/activate  
```

### Step 3: Install Dependencies

Install the required dependencies using the requirements.txt file:

```{bash}
pip install -r requirements.txt
```

### Step 4: Load and Preprocess the Data

Ensure the taxi dataset is placed in the `data/` folder. The expected format is yellow_tripdata_2024-01.parquet. You can replace this with other parquet datasets if needed.

In the main script (train_model.py), data is loaded and preprocessed with the following steps:

+ Loading the raw data: Using raw_taxi_df from data_utils.py.
+ Cleaning the data: Using clean_taxi_data from data_utils.py.
+ Splitting the data: Using split_taxi_data to split the dataset into training and test sets.

### Step 5: Train the Model

Run the main script to train the model:

```{bash}
python train_model.py
```

The script will:

+ Load and clean the data.
+ Split the data into training and testing sets.
+ Train a multi-layer perceptron (MLP) model based on the architecture specified in model_utils.py.
+ Save the trained model in the models/ directory.
