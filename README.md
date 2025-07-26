# Abstract:

A dashboard for analyzing Steam game performance is presented in this project. It employs data to investigate significant elements such as game revenue, reviews, and long-term category trends. Visual tools like word clouds, pie charts, line graphs, and scatter plots are available on the dashboard. These resources aid in comprehending what contributes to the success of games. The findings demonstrate the most lucrative game categories, the impact of reviews on earnings, and the evolution of player preferences over time. Players, marketers, and game developers who wish to learn more about the gaming industry will find this work helpful. The results can be used to improve future game decisions and increase their success.

# Details: 

### Data Directory: contains both the .csv files for the project.

### Images Directory: contains the images plotted in the main.ipynb file

### Modules Directory:

Categories.py: contains the list of selected categories/tags for visualization.

Dashboard.py: contains the dashboard implementation.

DataCleaner.py: contains all the methods which help in the preprocessing of data.

DataIntegrator.py: contains the method to combine both dataframes.

Imputer.py: Implementation of xgboost and optuna for filling nan values for review_summary column.

### main.ipynb: The main file which executes the above files and displays the dashboard with charts.

## Please make sure that dash, optuna and xgboost libraries are installed before running the main.ipynb file

Install by: pip install dash optuna xgboost# Dynamic-Game-Data-Analytics-Platform-Using-Plotly-and-Advanced-Data-Imputation-Techniques
