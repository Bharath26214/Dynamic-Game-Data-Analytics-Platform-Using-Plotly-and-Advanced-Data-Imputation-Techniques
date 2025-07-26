import dash
from dash import Dash, dcc, html
import random

from modules.DataVisualizer import DataVisualizer
from modules.Categories import selected_categories

class DashBoard:
    """
    Creates a dashboard.
    """
    def __init__(self, df):
        self.df = df
        self.selected_categories = selected_categories
        self.visualizer = DataVisualizer(self.df, self.selected_categories)
        self.app = dash.Dash(__name__)
        self.app_layout()
        self.register_callbacks()  

    def app_layout(self):
        """
        Configures the layout of the Dash application with tabs and dynamic content.
        """
        self.app.layout = html.Div([
            html.H1("Analysis of Steam Games and their Categories", style={"textAlign": "center"}),
            dcc.Tabs(
                id='tabs', 
                value='Introduction',  
                children=[
                    dcc.Tab(label='Introduction', value='Introduction'),
                    dcc.Tab(label='Pie Chart Representing percentage of game reviews', value='pie_chart'),
                    dcc.Tab(label='Top Games by Revenue vs Top Games by Reviews', value='top_games_comparison'),
                    dcc.Tab(label='Revenue Collected by Top Categories Over Time', value='revenue_by_genre'),
                    dcc.Tab(label='Trends of Game Tags Over Certain Time Periods', value='line_plot'),
                    dcc.Tab(label='Analysis of Revenue Collected vs Review Score', value='bubble_chart'),
                    dcc.Tab(label='Number of Games Released and Revenue Earned Every Year', value='num_of_games_and_their_revenues'),
                ],
                style={"overflow": "hidden"},
                content_style={"padding": "10px"}
            ),
            html.Div(id='tab-content') 
        ])
        return None

    def register_callbacks(self):
        """
        Registers callbacks to dynamically update tab content based on the selected tab.
        """
        @self.app.callback(
            dash.dependencies.Output('tab-content', 'children'),  
            [dash.dependencies.Input('tabs', 'value')] 
        )
        def update_content(tab_name):
            """
            Returns the appropriate content for the selected tab.
            """
            if tab_name == 'Introduction':
                return self.visualizer.introduction
            elif tab_name == 'pie_chart':
                return self.visualizer.percentage_of_game_summary
            elif tab_name == 'top_games_comparison':
                return self.visualizer.top_games_comparison
            elif tab_name == 'revenue_by_genre':
                return self.visualizer.revenue_by_genre
            elif tab_name == 'line_plot':
                return self.visualizer.trends_game_tags
            elif tab_name == 'bubble_chart':
                return self.visualizer.reviews_vs_revenue_over_time
            elif tab_name == 'num_of_games_and_their_revenues':
                return self.visualizer.production_and_revenue_over_years
            else:
                return html.Div("Tab not found.")

    def run(self):
        """
        Starts the Dash application server on a random port.
        """
        self.app.run_server(debug=False, port=random.randint(8050, 9000))
        return None
