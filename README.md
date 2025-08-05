# Machine Learning Final Project

This is the final project of our Machine Learning bootcamp, where we demonstrate the skills and knowledge acquired throughout our studies. Throughout this bootcamp, we have studied different models based on projects of different areas and types. Now it's time to create our own project using the algorithm that we think is best suited to our problem.

We will have to find a suitable dataset to work with, process it, train a model and finally make it available for consumption.

> *"Hard work always beats talent when talent doesn't work hard"* - Tim Notke

## 👥  Credits

**Team Members:**
> - Member 1
> - Member 2
> - Member 3

**Academy:** 
> - [4Geeks Academy](https://4geeksacademy.com/us/index) 
> - **Bootcamp:** Spain-DS-17 
> - **Mentor:** [Ing. Héctor Chocobar Torrejón](https://github.com/hchocobar/)
> - **Teacher Assitant:** [Beatriz Solana Ros](https://github.com/mezcolantriz)

## 🎯 Project Goal

The goal of this project is to develop a complete end-to-end Machine Learning solution that includes:
- Data acquisition and processing
- Exploratory Data Analysis (EDA)
- Model development and optimization
- Web application deployment
- Real-world problem-solving through ML techniques

## 🚀 Project Overview

*[This section will be updated as we define our specific project scope and objectives]*

### Problem Statement
*To be defined - we will identify a real-world problem that can be solved using Machine Learning techniques*

### Dataset
*To be defined - we will acquire a dataset that meets the following minimum requirements:*
- 60,000+ instances (rows)
- 20+ predictor variables (including at least 1 categorical variable)

### Methodology
*To be defined - we will document our chosen approach and algorithms*

### Results
*To be updated with our findings and model performance*

## 📝 Project Phases

### Step 1: Problem Definition
Start by defining a problem and turn it into a Machine Learning problem. This is the first step, since the data must meet a certain need and the Machine Learning process must aim at satisfying that need.

The choice of the data set must satisfy minimum requirements in terms of number of rows and predictor variables. At a minimum, it must contain:
- 60,000 instances (rows)
- 20 predictor variables, of which there must be at least 1 categorical variable

**NOTE:** Depending on the dataset and the case study to be explored, datasets that do not reach the established minimum may be evaluated and accepted.

### Step 2: Acquiring and Loading the Data Set
Since in the real world data does not usually arrive in a flat csv file, this data must be acquired by one of the following ways:
- Extracting data from some web page or portal using web scraping techniques
- Exploitation of a public database using SQL language (the database must support this language)
- Exploitation of a public API to obtain data

Once you have the data, you must store it in a CSV document and load it into Python using Pandas.

**NOTE:** Depending on the dataset and the case study to be explored, datasets downloaded by other means could be evaluated and accepted.

### Step 3: Store the Information
A widely used practice is to store the data, especially if they are massive, in a database for quick access to them. From all the databases we have studied, choose the one most compatible with your data and store it there. Then, perform queries using Python (with pure SQL code or using the wrappers we have studied in the course) to use the different statements: SELECT, JOIN, INSERT.... These queries must provide a value to start the analysis on the data prior to the statistics and EDA.

It is important to understand that in the real world we do not only have CSV as an ally to store data, since it is easier to lose a flat file like CSV than a database with its connections and data models inside. Security is also a critical and important factor for storing your data there, since a CSV does not provide any protection mechanism that other technologies do.

### Step 4: Perform a Descriptive Analysis
The raw data stored in a database can be a great and very valuable source of information. Before we begin to simplify and exploit them with EDA, we must know their fundamental statistical measures: means, modes, distributions, deviations, etcetera. Analyze the descriptive statistical variables of each of the predictors of the data set and theorize about the distribution that each of them follows.

Use hypothesis tests if you consider it necessary.

### Step 5: Perform a Full EDA
This step is vital to ensure that we keep the variables that are strictly necessary and eliminate those that are not relevant or do not provide information. Use the example Notebook we worked on and adapt it to this use case.

Make sure to conveniently divide the data set into train and test as we have seen in previous lessons.

### Step 6: Build the Model and Optimize It
Once you have your data ready, decide which model fits best and train it. If in doubt, try using several of the ones you have already studied. Select the one that best fits the data.

Remember that the hyperparameter optimization step is very important to explore and achieve the best version of the model.

### Step 7: Deploy the Model
Create a Machine Learning web application using your saved model. You can use Flask, Streamlit or any other tool you know. Use Heroku, Render or another cloud computing platform of your choice to deploy your web application and share it with the world. Remember that the application is going to be the gateway to potential users or customers, and you have to take care of even the smallest detail.

## 📁 Project Structure

```
ml-project-repo/
├── 📁 data/                # Raw and processed datasets
│    ├── 📁 interin/        # For intermediate data that has been transformed.
│    ├── 📁 processed/      # For the final data to be used for modeling.
│    ├── 📁 raw/            # For raw data without any processing.
├── 📁 database/            # SQL scripts and database configs
├── 📁 docs/                # Documentation and presentation materials
├── 📁 models/              # Trained model artifacts
├── 📁 notebooks/           # Jupyter notebooks for EDA and analysis
├── 📁 src/                 # Source code modules
├── 📁 webapp/              # Flask/Streamlit application
```

## 🛠️ Technologies Used

*[To be updated as we select our tech stack]*

## 📊 Results

*[To be updated with our model performance and insights]*

## 🌐 Live Demo

*[Link to be added when the web application is deployed]*
