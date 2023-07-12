#!/usr/bin/env python
# coding: utf-8

# # Activity: Structure your data 

# ## Introduction
# 
# In this activity, you will practice structuring, an **exploratory data analysis (EDA)** step that helps data science projects move forward. During EDA, when working with data that contains aspects of date and time, "datetime" transformations are integral to better understanding the data. As a data professional, you will encounter datatime transformations quite often as you determine how to format your data to suit the problems you want to solve or the questions you want to answer. This activity gives you an opportunity to apply these skills and prepare you for future EDA, where you will need to determine how best to structure your data.
# 
# In this activity, you are a member of an analytics team that provides insights to an investing firm. To help them decide which companies to invest in next, the firm wants insights into **unicorn companies**–companies that are valued at over one billion dollars.  
# 
# You will work with a dataset about unicorn companies, discovering characteristics of the data, structuring the data in ways that will help you draw meaningful insights, and using visualizations to analyze the data. Ultimately, you will draw conclusions about what significant trends or patterns you find in the dataset. This will develop your skills in EDA and your knowledge of functions that allow you to structure data.
# 
# 
# 
# 

# ## Step 1: Imports 

# ### Import relevant libraries and modules
# 
# Import the relevant Python libraries and modules that you will need to use. In this activity, you will use `pandas`, `numpy`, `seaborn`, and `matplotlib.pyplot`.

# In[1]:


# Import the relevant Python libraries and modules needed in this lab.

import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 


# ### Load the dataset into a DataFrame
# 
# The dataset provided is in the form of a csv file named `Unicorn_Companies.csv` and contains a subset of data on unicorn companies. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ###
companies = pd.read_csv("Unicorn_Companies.csv")


# ## Step 2: Data exploration
# 

# ### Display the first 10 rows of the data
# 
# In this section, you will discover what the dataset entails and answer questions to guide your exploration and analysis of the data. This is an important step in EDA. 
# 
# To begin, display the first 10 rows of the data to get an understanding of how the dataset is structured. 

# In[3]:


# Display the first 10 rows of the data.

companies.head(10)


# ### Identify the number of rows and columns
# 
# Identify the number of rows and columns in the dataset. This will help you get a sense of how much data you are working with.

# In[5]:


# Identify the number of rows and columns in the dataset.


companies.shape


# ### Check for duplicates in the data

# In[6]:


# Check for duplicates.

# Assuming you have a DataFrame named 'companies'
original_shape = companies.shape

# Drop duplicates from the DataFrame
companies = companies.drop_duplicates()

# Compare the shape before and after dropping duplicates
new_shape = companies.shape

# Print the original and new shape
print("Original Shape:", original_shape)
print("New Shape:", new_shape)


# No dupliactes in the values
# 

# ### Display the data types of the columns 
# 
# Knowing the data types of the columns is helpful because it indicates what types of analysis and aggregation can be done, how a column can be transformed to suit specific tasks, and so on. Display the data types of the columns. 

# In[11]:


# Display the data types of the columns.

companies.dtypes


# ### Sort the data
# 
# In this section, you will continue your exploratory data analysis by structuring the data. This is an important step in EDA, as it allows you to glean valuable and interesting insights about the data afterwards.
# 
# To begin, sort the data so that you can get insights about when the companies were founded. Consider whether it would make sense to sort in ascending or descending order based on what you would like to find.

# In[16]:


# Sort `companies` and display the first 10 rows of the resulting DataFrame.

# Sort the DataFrame by 'year founded' in descending order
companies_sorted = companies.sort_values(by='Year Founded', ascending=False)
companies_sorted.head (10)


# ### Determine the number of companies founded each year
# 
# Find out how many companies in this dataset were founded each year. Make sure to display each unique `Year Founded` that occurs in the dataset, and for each year, a number that represents how many companies were founded then.

# In[17]:


# Display each unique year that occurs in the dataset
# along with the number of companies that were founded in each unique year.

company_counts = companies['Year Founded'].value_counts()

# Print the number of companies founded each year
print(company_counts)


# **2015 has the highest number of companies founded. 

# Plot a histogram of the `Year Founded` feature.

# In[18]:


# Plot a histogram of the Year Founded feature.
plt.hist(companies['Year Founded'], bins=30)
plt.xlabel('Year Founded')
plt.ylabel('Count')
plt.title('Histogram of Year Founded')
plt.show()


# **Question:** If you want to compare when one company joined unicorn status to when another company joined, how would you transform the `Date Joined` column to gain that insight? To answer this question, notice the data types.
# 

# ### Convert the `Date Joined` column to datetime
# 
# Convert the `Date Joined` column to datetime. This will split each value into year, month, and date components, allowing you to later gain insights about when a company gained unicorn status with respect to each component.

# In[23]:


# Convert the `Date Joined` column to datetime.
# Update the column with the converted values.

companies_new = companies.copy()  # Create a copy of the DataFrame
companies_new['Date Joined'] = pd.to_datetime(companies_new['Date Joined'])

# Display the data types of the columns in the new DataFrame
print(companies_new.dtypes)
print(companies.dtypes)


# In[29]:


## Extract Year Joined form Date Joined.

companies_new['Year Joined'] = pd.to_datetime(companies_new['Date Joined']).dt.year


# Print the DataFrame with the new 'Year Joined' column
companies_new.head(10)


# In[30]:


# Display each unique year which companies joined unicorn status that occurs in the dataset

company_unicorn_count = companies_new['Year Joined'].value_counts()

# Print the number of companies founded each year
print(company_unicorn_count)


# In[31]:


# Obtain the names of the months when companies gained unicorn status.
# Use the result to create a `Month Joined` column.

companies_new['Month Joined'] = pd.to_datetime(companies_new['Date Joined']).dt.month

company_unicorn_Month_Count = companies_new['Month Joined'].value_counts()

# Display the first few rows of `companies`
# to confirm that the new column did get added.
# Print the DataFrame with the new 'Month Joined' column
print(company_unicorn_Month_Count)


# In[32]:


companies_new['Month Joined'] = pd.to_datetime(companies_new['Date Joined']).dt.strftime('%b')

# Print the DataFrame with the new 'Month Joined' column
companies_new.head(20)


# In[33]:


company_unicorn_Month_Count = companies_new['Month Joined'].value_counts()
print(company_unicorn_Month_Count)


# In[39]:


# Plot a histogram of the Year Founded feature.
months_ordered = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


plt.hist(companies_new['Month Joined'], bins=12, edgecolor='black')
plt.xlabel('Month Joined')
plt.ylabel('Count')
plt.title('Histogram of Date Joined')
plt.xticks(months_ordered)
plt.show()


# ### Create a `Years To Join` column
# 
# Determine how many years it took for companies to reach unicorn status, and use the result to create a `Years To Join` column. Adding this to the dataset can help you answer questions you may have about this aspect of the companies.

# In[43]:


# Determine how many years it took for companies to reach unicorn status.
# Use the result to create a `Years To Join` column.



companies_new['Years to Unicorn'] = companies_new['Year Joined'] - companies_new['Year Founded']

# Print the DataFrame with the new 'Years to Unicorn' column

companies_new.head(10)


# In[48]:


# Plot a histogram of the Year to join.
plt.hist(companies_new['Years to Unicorn'], bins=30)
plt.xlabel('Year to Unicorn')
plt.ylabel('Count')
plt.title('Histogram of Years to Unicorn')
plt.show()


# In[54]:


plt.boxplot(x=companies_new['Years to Unicorn'])
plt.title('Box Plot of Years to Join')
plt.show()


# In[46]:


mean_years_to_unicorn = companies_new['Years to Unicorn'].mean()
mode_years_to_unicorn = companies_new['Years to Unicorn'].mode().values[0]
median_years_to_unicorn = companies_new['Years to Unicorn'].median()

print("Mean Years to Unicorn:", mean_years_to_unicorn)
print("Mode Years to Unicorn:", mode_years_to_unicorn)
print("Median Years to Unicorn:",median_years_to_unicorn)


# ### Gain more insight on a specific year
# 
# To gain more insight on the year of that interests you, filter the dataset by that year and save the resulting subset into a new variable. 

# In[57]:


# Filter dataset by a year of your interest (in terms of when companies reached unicorn status).
# Save the resulting subset in a new variable. 

joined_2015 = companies_new[companies_new['Year Joined'] == 2015]


# Display the first few rows of the subset to confirm that it was created.

print(joined_2015)


# In[58]:



# Determine the most common industry for companies joined in 2015
most_common_industry = joined_2015['Industry'].mode().values[0]

print("Most Common Industry for Companies Joined in 2015:", most_common_industry)


# In[60]:


top_3_industries = joined_2015['Industry'].value_counts().head(3).index.tolist()
print("Most Common Industry for Companies Joined in 2015:", top_3_industries)


# In[82]:


companies_new['Valuation (in Billions)'] = companies_new['Valuation'].str.extract('(\d+\.?\d*)B', expand=False).astype(float)
#companies_new['Funding (in Billions)'] = companies_new['Funding'].str.extract('(\d+\.?\d*)M','(\d+\.?\d*)B', expand=False).astype(float)
#companies_new['Funding (in Billions)'] = companies_new['Funding'].str.extract('(\d+\.?\d*)B', expand=False).astype(float)

#companies_new['Funding'].str.extract('(\d+\.?\d*)M', expand=False).astype(float)
companies_new['Funding'].str.extract('(\d+\.?\d*)B', expand=False).astype(float)

#companies['Funding (in Millions)'] = funding_millions
#companies['Funding (in Billions)'] = funding_billions * 1000

print
companies_new.head(10)
#companies_new.describe
#top_5_companies = companies_new.groupby('Industry').agg({'Valuation (in Billions)': 'sum', 'Funding': 'sum'}).nlargest(5, 'Valuation (in Billions)')

#top_5_companies.head(10)


# In[62]:


top_5_companies = companies_new.groupby('Industry').agg({'Valuation': 'sum', 'Funding': 'sum'}).nlargest(5, 'Valuation')

print(top_5_companies)


# In[ ]:





# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about structuring data in Python](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/Vh13u/use-structuring-methods-to-establish-order-in-your-dataset).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the property in the `pandas` library that contains datetime strings in order to extract the year components.
# 
# Use square brackets to filter a DataFrame in order get a subset of the data. Make sure to specify an appropriate condition inside those brackets. The condition should convey which year you want to filter by. The rows that meet the condition are the rows that will be selected.
# 
# Use the function in the `pandas` library that allows you to display the first few rows of a DataFrame.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `dt.year` property on the `Date Joined` column to obtain the years that companies became unicorns.
# 
# Make sure to create a new variable and assign it to the subset. 
# 
# Use the `head()` function to display the first few rows of a DataFrame.
# 
# </details>

# **Question:** Using a time interval, how could you observe trends in the companies that became unicorns in one year?
# 

# [Write your response here. Double-click (or enter) to edit.]

# ### Observe trends over time
# 
# Implement the structuring approach that you have identified to observe trends over time in the companies that became unicorns for the year that interests you.

# In[ ]:


# After identifying the time interval that interests you, proceed with the following:
# Step 1. Take the subset that you defined for the year of interest. 
#         Insert a column that contains the time interval that each data point belongs to, as needed.
# Step 2. Group by the time interval.
#         Aggregate by counting companies that joined per interval of that year.
#         Save the resulting DataFrame in a new variable.

### YOUR CODE HERE ###





# Display the first few rows of the new DataFrame to confirm that it was created

### YOUR CODE HERE ###



# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about structuring data in Python](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/Vh13u/use-structuring-methods-to-establish-order-in-your-dataset).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# To obtain the data in a specific periodic datetime format, call a function in the `pandas` library on a series that contains datetime strings.   
# 
# Keep in mind that currently, the `Valuation` column is of data type `object` and contains `$` and `B` to indicate that each amount is in billions of dollars.
# 
# Call functions in the `pandas` library to achieve the following tasks:
#   - Apply a function to each value in the series.
#   - Cast each value in the series to a specified data type.
# 
# Use a pair of square brackets to access a particular column from the result of grouping a DataFrame. 
# 
# Use these functions in the `pandas` library to achieve the following tasks:
# - Concatenate two DataFrames together
# - Drop columns that you do not need from a DataFrame
# - Group a DataFrame by a specific column
# - Compute the average value for each group
# - Reset the index so that the column that you grouped on also appears as a column after the grouping (instead of remaining an index) 
# - Rename columns in a DataFrame
# - Display the first few rows of a DataFrame
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `dt.strftime('%Y-W%V')` on the `Date Joined` column to obtain the weeks that companies became unicorns.
# 
# Use these functions in `pandas` to acheive the following tasks:
# - `groupby()` to group a DataFrame by a specific column
# - `count()` to count the number of rows that belong to each group
# - `reset_index()` to reset the index so that the column that you grouped on also appears as a column after the grouping (instead of remaining an index) 
# - `rename()` to rename the columns in a DataFrame
# - `head()` to display the first few rows of a DataFrame
# 
# </details>

# **Question:** How would you structure the data to observe trends in the average valuation of companies from 2020 to 2021?  

# [Write your response here. Double-click (or enter) to edit.]

# ### Compare trends over time
# 
# Implement the structuring approach that you have identified in order to compare trends over time in the average valuation of companies that became unicorns between your years of interest. Keep in mind the data type of the `Valuation` column and what the values in that column contain currently.

# In[ ]:


# After identifying the additional year and time interval of interest, proceed with the following:
# Step 1. Filter by the additional year to create a subset that consists of companies that joined in that year.
# Step 2. Concatenate that new subset with the subset that you defined previously.
# Step 3. As needed, add a column that contains the time interval that each data point belongs to, 
#         in the concatenated DataFrame.
# Step 4. Transform the `Valuation` column as needed.
# Step 5. Group by the time interval.
#         Aggregate by computing average valuation of companies that joined per interval of the corresponding year.
#         Save the resulting DataFrame in a new variable.

### YOUR CODE HERE ###



# Display the first few rows of the new DataFrame to confirm that it was created.

### YOUR CODE HERE ###



# 
# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about manipulating data in Python](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/0GjvK/date-string-manipulations-with-python).
# 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# To obtain the data in a specific periodic datetime format, call a function in the `pandas` library on a series that contains datetime strings.   
# 
# Keep in mind that currently, the `Valuation` column is of data type `object` and contains `$` and `B` to indicate that each amount is in billions of dollars.
# 
# Call functions in the `pandas` library on a series to acheive the following tasks:
#   - Apply a function to each value in the series.
#   - Cast each value in the series to a specified data type.
# 
# Use a pair of square brackets to access a particular column from the result of grouping a DataFrame. 
# 
# These functions in the `pandas` library can help achieve the following tasks:
# - Concatenate two DataFrames together
# - Drop columns that you do not need from a DataFrame
# - Group a DataFrame by a specific column
# - Compute the average value for each group
# - Reset the index so that the column that you grouped on also appears as a column after the grouping (instead of remaining an index) 
# - Rename columns in a DataFrame
# - Display the first few rows of a DataFrame
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `.dt.to_period('Q').dt.strftime('%Y-Q%q')` on the `Date Joined` column to obtain the quarters during which companies became unicorns.
# 
# Convert the `Valuation` column to numeric by removing `$` and `B` and casting each value to data type `float`.
# 
# Use the following functions in `pandas` to acheive the following tasks:
# - `concat` to concatenate two DataFrames together (note: this function takes in a list of DataFrames and returns a DataFrame that contains all rows from both inputs)
# - `drop()` to drop columns that you do not need from a DataFrame
# - `groupby()` to group a DataFrame by a specific column
# - `mean()` to compute the average value for each group
# - `reset_index()` to reset the index so that the column that you grouped on also appears as a column after the grouping (instead of remaining an index) 
# - `rename()` to rename the columns in a DataFrame
# - `head()` to display the first few rows of a DataFrame
# 
# </details>

# ## Step 3: Statistical tests
# 
# ### Visualize the time it took companies to become unicorns
# 
# Using the `companies` dataset, create a box plot to visualize the distribution of how long it took companies to become unicorns, with respect to the month they joined. 

# In[ ]:


# Define a list that contains months in chronological order.

### YOUR CODE HERE ###


# Print out the list to confirm it is correct.

### YOUR CODE HERE ###


            


# In[ ]:


# Create the box plot to visualize the distribution of how long it took companies to become unicorns, with respect to the month they joined.
# Make sure the x-axis goes in chronological order by month, using the list you defined previously.
# Plot the data from the `companies` DataFrame.

### YOUR CODE HERE ###



# Set the title of the plot.

### YOUR CODE HERE ###



# Rotate labels on the x-axis as a way to avoid overlap in the positions of the text.  

### YOUR CODE HERE ###



# Display the plot.

### YOUR CODE HERE ###



# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about creating a box plot](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/Pf6KW/eda-structuring-with-python).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a box plot.
# 
# Use the functions in the `matplotlib.pyplot` module that allow you to acheive the following tasks:
# - set the title of a plot
# - rotate labels on the x-axis of a plot
# - display a plot
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `boxplot()` function from `seaborn` to create a box plot, passing in the parameters `x`, `y`, `order`, and `showfliers`. To keep outliers from appearing on the box plot, set `showfliers` to `False`.
# 
# Use following functions to achieve the following tasks:
# - `plt.title()` to set the title of a plot
# - `plt.xticks()` to rotate labels on the x-axis of a plot
# - pass in the parameters `rotation=45, horizontalalignment='right'`to rotate the labels by 45 degrees and align the labels to the right
# - `plt.show()` to display a plot
# 
# </details>

# **Question:** In the preceding box plot, what do you observe about the median value for `Years To Join` for each month?
# 

# [Write your response here. Double-click (or enter) to edit.]

# ## Step 4: Results and evaluation
# 

# ### Visualize the time it took companies to reach unicorn status
# 
# In this section, you will evaluate the result of structuring the data, making observations, and gaining further insights about the data. 
# 
# Using the `companies` dataset, create a bar plot to visualize the average number of years it took companies to reach unicorn status with respect to when they were founded. 

# In[ ]:


# Set the size of the plot.

### YOUR CODE HERE ###




# Create bar plot to visualize the average number of years it took companies to reach unicorn status 
# with respect to when they were founded.
# Plot data from the `companies` DataFrame.

### YOUR CODE HERE ###




# Set title

### YOUR CODE HERE ###




# Set x-axis label

### YOUR CODE HERE ###




# Set y-axis label

### YOUR CODE HERE ###




# Rotate the labels on the x-axis as a way to avoid overlap in the positions of the text.  

### YOUR CODE HERE ###



# Display the plot.

### YOUR CODE HERE ###


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about creating a bar plot](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/0GjvK/date-string-manipulations-with-python).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a bar plot where the height of each bar is the average value for the corresponding category, by default.
# 
# Use the functions in the `matplotlib.pyplot` module that allow you to set the size, title, x-axis label, and y-axis label of plots. In that module, there are also functions for rotating the labels on the x-axis and displaying the plot. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `barplot()` function from `seaborn`, passing in the parameters `x`, `y`, and `ci`. To keep confidence interval lines from appearing on the bar plot,  set `ci` to `False`.
# 
# Use `plt.figure()`, passing in the `figsize` parameter to set the size of a plot.
# 
# Use `plt.title()`, `plt.xlabel()`, `plt.ylabel()` to set the title, x-axis label, and y-axis label, respectively. 
# 
# Use `plt.xticks()` to rotate labels on the x-axis of a plot. Paass in the parameters `rotation=45, horizontalalignment='right'` to rotate the labels by 45 degrees and align the labels to the right.
# 
# Use `plt.show()` to display a plot.
# 
# </details>

# **Question:** What trends do you notice in the data? Specifically, consider companies that were founded later on. How long did it take those companies to reach unicorn status?
# 

# [Write your response here. Double-click (or enter) to edit.]

# ### Visualize the number of companies that joined per interval 
# 
# Using the subset of companies joined in the year of interest, grouped by the time interval of your choice, create a bar plot to visualize the number of companies that joined per interval for that year. 

# In[ ]:


# Set the size of the plot.

### YOUR CODE HERE ###



# Create bar plot to visualize number of companies that joined per interval for the year of interest.

### YOUR CODE HERE ###



# Set the x-axis label.

### YOUR CODE HERE ###



# Set the y-axis label.

### YOUR CODE HERE ###



# Set the title.

### YOUR CODE HERE ###



# Rotate labels on the x-axis as a way to avoid overlap in the positions of the text.  

### YOUR CODE HERE ###



# Display the plot.

### YOUR CODE HERE ###


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about creating a bar plot](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/0GjvK/date-string-manipulations-with-python).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a bar plot where the height of each bar is the average value for the corresponding category, by default.
# 
# Use the functions in the `matplotlib.pyplot` module that allow you to set the size, title, x-axis label, and y-axis label of plots. In that module, there are also functions for rotating the labels on the x-axis and displaying the plot. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `barplot()` function from `seaborn`, passing in the parameters `x`, `y`, and `ci`. To keep confidence interval lines from appearing on the bar plot,  set `ci` to `False`.
# 
# Use `plt.figure()`, passing in the `figsize` parameter to set the size of a plot.
# 
# Use `plt.title()`, `plt.xlabel()`, `plt.ylabel()` to set the title, x-axis label, and y-axis label, respectively. 
# 
# Use `plt.xticks()` to rotate labels on the x-axis of a plot. Paass in the parameters `rotation=45, horizontalalignment='right'` to rotate the labels by 45 degrees and align the labels to the right.
# 
# Use `plt.show()` to display a plot.
# 
# </details>

# **Question:** What do you observe from the bar plot of the number of companies that joined per interval for the year of 2021? When did the highest number of companies reach $1 billion valuation?
# 
#   

# [Write your response here. Double-click (or enter) to edit.]

# ### Visualize the average valuation over the quarters
# 
# Using the subset of companies that joined in the years of interest, create a grouped bar plot to visualize the average valuation over the quarters, with two bars for each time interval. There will be two bars for each time interval. This allows you to compare quarterly values between the two years.

# In[ ]:


# Using slicing, extract the year component and the time interval that you specified, 
# and save them by adding two new columns into the subset. 

### YOUR CODE HERE ###



# Set the size of the plot.

### YOUR CODE HERE ###



# Create a grouped bar plot.

### YOUR CODE HERE ###



# Set the x-axis label.

### YOUR CODE HERE ###



# Set the y-axis label.

### YOUR CODE HERE ###



# Set the title.

### YOUR CODE HERE ###



# Display the plot.

### YOUR CODE HERE ###


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about creating a grouped bar plot](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/Pf6KW/eda-structuring-with-python).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a grouped bar plot, specifying the category and height for each bar, as well as the hue.
# 
# Use the functions in the `matplotlib.pyplot` module that allow you to set the size, title, x-axis label, and y-axis label of plots. In that module, there is also a function for displaying the plot. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `plt.bar()` to create the bar plot, passing in the parameters `x`, `y`, and `hue`. For the task at hand, set `hue` to the column that contains year joined. 
# 
# Use `plt.figure()`, passing in the `figsize` parameter to set the size of a plot.
# 
# Use `plt.title()`, `plt.xlabel()`, `plt.ylabel()` to set the title, x-axis label, and y-axis label, respectively. 
# 
# Use `plt.show()` to display a plot.
# 
# </details>

# **Question:** What do you observe from the preceding grouped bar plot?
# 
#   

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** Is there any bias in the data that could potentially inform your analysis?
# 

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What potential next steps could you take with your EDA?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** Are there any unanswered questions you have about the data? If yes, what are they?
# 

# [Write your response here. Double-click (or enter) to edit.]

# ## Considerations

# **What are some key takeaways that you learned from this lab?**

# [Write your response here. Double-click (or enter) to edit.]

# **What findings would you share with others?**

# [Write your response here. Double-click (or enter) to edit.]

# **What recommendations would you share with stakeholders based on these findings?**

# [Write your response here. Double-click (or enter) to edit.]

# **References**
# 
# Bhat, M.A. (2022, March).[*Unicorn Companies*](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies). 
# 
# 
