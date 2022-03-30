# Project Description
The scenario for this project assumes that I am a new junior data scientist on the Zillow data science team. The team has already built a model to predict property tax assessed values of Single Family Properties that had a transaction in 2017. They would like me to provide my own insights and see if I can build a better regression model that improves upon their work.

## Goals/deliverables
- Construct an ML Regression model that predict propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.
- Find the key drivers of property value for single family properties. Some questions that come to mind are: Why do some properties have a much higher value than others when they are located so close to each other? Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location? Is having 1 bathroom worse than having 2 bedrooms?
- Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
- Make recommendations on what works or doesn't work in prediction these homes' values.

## Planning Steps:
- Create a README.md to document my process, my key findings and provide instructions on how to replicate the project.
- Acquire the zillow data from the MySQL database and then create and store functions to replicate this process in a wrangle.py file.
- Complete initial summary of data to see what I am working with: .info(), .describe(), .value_counts()
- Decided which columns are useful and relevant to move forward with. 
- Conduct intial univariate exploration, plot distributions of relevant variables, and make decisions on which variables I will move forward with for further exploration and eventually modeling.
    - I will begin first iteration by creating a dataframe that only contains the required variables to reach the Minimum Viable Product: square feet, number of bedrooms and bathrooms, and the target -- taxvaluedollarcnt.
- Conduct further cleaning and preparation of data: 
    - Dropping outliers if necessary
    - Removing or imputing nulls if necessary
    - Renaming columns
    - Changing data types
- Create a seperate data frame that converts fips to corresponding county for exploration. 
- Form initial hypotheses and questions to investigate
- Split data and begin further exploration on the train data.
    - Evaluate hypotheses using statistical tests
    - Create visualizations of variable interactions
    - Decide which variables are key drivers to be used in modeling
- Split data into X and y subgroup to be used in modeling
- Set a baseline using the target mean or median. 
- Conduct preprocessing including one hot encoding and creating dummy variables where necessary before modeling.
- Scale data if necesary.
- Create and fit models on train data set and then evaluate on the validate data set. 
- Choose my best model and evaluate it on the test data set. 
- Document my findings and conclusions in a Final Report Notebook.

# Hypotheses

1. There is a linear relationship between a home's number of bedrooms and it's assessed value
- Null Hypothesis Rejected
2. There is a linear relationship between a home's number of bathrooms and it's assessed value
- Null Hypothesis Rejected
3. Homes have a higher assessed value based on what county they are in.
- Homes in Ventura and Orange County are valued higher than homes in LA County
4. The year a home was built (it's age) is linearly correlated with it's assessed value
- The newer a home the higher it is valued


## Data Dictionary

| Feature           | Datatype   | Definition                                 |    
|:------------------|:-----------|:-------------------------------------------|
| bedrooms          | int64      | Number of bedrooms in the property         |
| bathrooms         | float64    | Number of bathrooms in the property        |
| year_built        | object     | Year that the property was built           |
| square_feet       | float64    | Total size of the property in square feet  |
| county            | object     | County where the property is located       |
| assessed_value    | float64    | Tax assessed value of the property         |


# Instructions to Replicate My Work
1. Clone this repository
2. Add your own env.py file for server credentials
3. Run the cells in final_notebook.ipynb