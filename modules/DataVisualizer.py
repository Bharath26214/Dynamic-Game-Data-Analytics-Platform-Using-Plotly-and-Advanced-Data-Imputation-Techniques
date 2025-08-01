import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dash import dcc, html
from wordcloud import WordCloud
import numpy as np

class DataVisualizer:
    """
    A class for visualizing data using Dash and Plotly.

    Attributes:
        df: The dataframe containing the data for visualization.
        selected_tags: A list of selected tags for filtering visualizations.
    """

    def __init__(self, df, selected_tags):
        """
        Initializes the DataVisualizer instance with a dataframe and selected tags.

        Args:
            df: The dataframe containing the data for visualization.
            selected_tags: A list of selected tags for filtering visualizations.
        """
        self.df = df
        self.df['Release Date'] = pd.to_datetime(self.df['Release Date'])
        self.selected_tags = selected_tags
        self.height = 800

    @property
    def introduction(self):
        """
        Creates an introduction for the dashboard summarizing insights from each tab.

        Returns:
            The intro for the dashboard as HTML content.
        """

        # creating word cloud to display in dashboard
        categories_string = ','.join(self.df['Tags'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(categories_string)

        wc_array = np.array(wordcloud)

        fig = px.imshow(wc_array)

        fig.update_layout(
            title_x=0.5,
            xaxis_visible=False,
            yaxis_visible=False,
            coloraxis_showscale=False
        )

        fig.update_traces(hoverinfo='skip')

        return html.Div([
            html.P("This dashboard offers a comprehensive analysis of Steam games across various dimensions, "
                    "including revenue, user reviews, and trends in game categories."),
            html.H2("Word cloud which shows the majority of game categories present in data", style={"marginTop": "20px"}),
            dcc.Graph(figure=fig),
            html.H2("Dashboard Tabs:", style={"marginTop": "20px"}),
            html.Ul([
                html.Li(["Pie Chart Representing Percentage of Game Reviews:", 
                        html.P('The pie chart represents the proportion of overall game outcome whether it is a success or a failure.'),
                        html.Ol([
                            html.Li('Majority of the games are reviewed as Positive and 541 games performed extradinaryly well.'),
                            html.Li('There are 31 percentage of games with an average rating and mixed reviews.'),
                            html.Li('Four percent (2000 games approx.) of games performed poorly due to the fact that it received maority of negative reviews.'),
                            ]),
                        ]),
        
                html.Li(["Top Games by Revenue vs. Top Games by Reviews:",
                        html.P('The Bar charts shows the Top 10 games which have collected the most revenue and which have the most number of user reviews.'),
                        html.Ol([
                            html.Li('Counter-Strike is the only game to have collected more than 100 Million USD followed by PUBG and Dota 2 with 66M USD and 60M USD respectively.'),
                            html.Li('The above mentioned games are the ones which have the most number of reviews as these features are higly correlated.'),
                            ]),
                        ]),

                html.Li(["Revenue Collected by Top Categories Over Time:",
                        html.P('The bar chart shows the Top tags which have collected the most revenue for more than 2 decades. The four pie charts represent the 4 time periods of what tags got popularized and what was their share out of the total revenue in that time period.'),
                        html.Ol([
                            html.Li('Co Op games have a market share of 2 Billion USD followed by action, first person, singleplayer and multiplayer games.'),
                            html.Li('The revenue generated by games from 2000 to 2023 increased significantly from 283 Million USD to 21 Billion USD.'),
                            html.Li('The tags which are present in the Bar Chart appear more frequently in the pie charts due to the fact that those were the tags which generated the highest revenue.'),
                            html.Li('Classic, Masterpiece games diminished in the future years while action, singleplayer, adventure games retained their popularity over the years.'),
                            ]),
                        ]),
                        
                html.Li(["Trends of Game Tags Over Time:",
                        html.P('The line charts represent the weighted average review score of selected game categories over time.'),
                        html.Ol([
                            html.Li('Majority of Tags such as Indie, Action maintained a consistancy while categories like Artificial Intelligence, 360 Video showed a rapid increase in the reviews showcasing the interest of gamers dwelling towards such games.'),
                            html.Li('Trading Card games also shown an increase from 73 in 2000-03 to 86 in 2020-23.'),
                            html.Li('Sci Fi and Story Rich games had a small bu visible decline in their reviews but this decline is not gradual'),
                            ]),
                        ]),

                html.Li(["Analysis of Revenue collected vs Review score",
                        html.P('The bubble charts shows games which have collected more than 5 Million USD along with 4 main features- year in which the game was released, review score, game outcome and estimated revenue'),
                        html.Ol([
                            html.Li('There is only one negatively reviewed game which has collected more than 5M USD which is Battle field in 2021.'),
                            html.Li('In the early years games which are reviewed as Overwhelmingly Positive are the only games which could collect over 5M USD but later games with a decent reviews were also able to collect good revenue.'),
                            html.Li('Until 2010 there are only 2 games- Garrys mod and Left 4 Dead 2 to have collected good revenue. In the later years a lot of other games joined the list with incredible collections especially Counter strike, Pubg and Dota 2.'),
                            ]),
                        ]),
                html.Li(["Number of Games Released and Revenue Earned Every Year: ",
                        html.P('The bar chart shows the number of games released in each year from 2000-23 and the line plot shows the revenue collected by all the games which are released in that particular year.'),
                        html.Ol([
                            html.Li('The number of games released increased drastically from 44 in 2000 to 2k in 2023 being at its peak in 2018.'),
                            html.Li('It is quite clear that the revenue collected will also increase as the number of games released have increased.'),
                            html.Li('The highest revenue generated was in the year 2015 which is 232 Million USD. This figure can be supported from the bubble chart where there is a lot of scatter points in the year 2015 thanks to games like GTAV, Call of Duty, Toms Clancy and a few more.'),
                            html.Li('The second highest revenue was collected in year 2020 which was 221 Million USD. Can be justified due to the fact that it was the Pandemic year where people had a lot of leisure time.'),
                            ]),
                        ])
            ]),
            html.P("Use the tabs above to navigate through these insights and explore the data in detail. We hope you enjoy the Dashboard!"),
        ])

    @property
    def trends_game_tags(self):
        """
        Creates a line plot showing trends of weighted average review scores for selected tags over time.

        Returns:
            A Dash Graph object displaying the trends.
        """
        self.df['weighted_review_score'] = self.df['review_score'] * self.df['Reviews Total']
        self.df['Time Period'] = pd.cut(self.df['Release Date'].dt.year,
                                bins=[2000, 2004, 2008, 2012, 2016, 2020, 2024],
                                labels=['2000-2003', '2004-2007','2008-2011', '2012-2015', '2016-2019', '2020-2023'],
                                right=False)

        df_exploded = self.df.assign(Tags=self.df['Tags'].str.split(',')).explode('Tags')
        df_exploded['Tags'] = df_exploded['Tags'].str.strip()
        filtered_tags = df_exploded[df_exploded['Tags'].isin(self.selected_tags)]

        tag_time_scores = filtered_tags.groupby(['Time Period', 'Tags']).apply(
            lambda x: pd.Series({
                'Weighted Average Review Score': (x['weighted_review_score'].sum() / x['Reviews Total'].sum()),
            })
        ).unstack()

        tag_time_scores.columns = tag_time_scores.columns.droplevel()

        fig = go.Figure()
        for tag in tag_time_scores.columns:
            fig.add_trace(go.Scatter(
                x=tag_time_scores.index,  
                y=tag_time_scores[tag],
                mode='lines+markers',
                name=tag,
                line=dict(width=2),
                marker=dict(size=6)        
            ))

        fig.update_layout(
            title='Weighted Average <b>Review Scores</b> of <b>Tags</b> Over Years:',
            xaxis_title='Time Period',
            height=self.height,
            yaxis_title='Weighted Average Review Score',
            legend_title='Tags',
            xaxis=dict(tickmode='array', tickvals=tag_time_scores.index, ticktext=tag_time_scores.index),
            yaxis=dict(showgrid=True),
            margin=dict(r=200),
            hovermode='closest'
        )

        return dcc.Graph(figure=fig)

    @property
    def percentage_of_game_summary(self):
        """
        Creates a pie chart showing the percentage of overall game outcomes.

        Returns:
            A Dash Graph object displaying the pie chart.
        """
        fig = go.Figure(data=go.Pie(values=self.df['review_summary'].value_counts().values,
                                    labels=self.df['review_summary'].value_counts().index,
                                    hole=0.3))

        fig.update_layout(title='Percentage of Overall <b>Game Outcome</b>:',
                          height=self.height,
                          legend_title='Review Summary',
                          hovermode='closest')

        return dcc.Graph(figure=fig)
    
    @property
    def revenue_by_genre(self):
        """
        Creates a combined layout showing:
        1. A bar chart for revenue collected by the top game categories.
        2. A grid of pie charts for revenue distribution across genres over time.

        Returns:
            A Dash HTML Div containing a single dcc.Graph with both visualizations.
        """
        def normalize_tag(tag):
            tag = tag.strip().lower()
            tag = tag.replace('co-op', 'co op')
            tag = tag.replace('first-person', 'first person')
            return tag

        copy_df = self.df.copy()
        copy_df['Release Year'] = copy_df['Release Date'].dt.year
        time_periods = [(2000, 2005), (2006, 2011), (2012, 2017), (2018, 2023)]

        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{'colspan': 2}, None],
                [{'type': 'domain'}, {'type': 'domain'}],
                [{'type': 'domain'}, {'type': 'domain'}] 
            ],
            subplot_titles=[
                "Top Categories by Revenue (Bar Chart)"
            ] + [f"{start}-{end}" for start, end in time_periods]
        )

        # getting data for the bar chart
        tag_revenue = self.df['Tags'].str.split(',').explode().reset_index()
        tag_revenue['Tags'] = tag_revenue['Tags'].apply(normalize_tag)
        tag_revenue['Revenue'] = self.df.loc[tag_revenue['index'], 'Revenue Estimated'].values
        tag_revenue = tag_revenue.groupby('Tags')['Revenue'].sum().reset_index()
        tag_revenue.columns = ['Genre', 'Revenue']
        tag_revenue = tag_revenue.sort_values(by='Revenue', ascending=False)

        fig.add_trace(
            go.Bar(
                x=tag_revenue['Genre'].head(20),
                y=tag_revenue['Revenue'].head(20),
                showlegend=False
            ),
            row=1, col=1
        )

        # getting pie charts data 
        for i, (start, end) in enumerate(time_periods):
            period_df = copy_df[(copy_df['Release Year'] >= start) & (copy_df['Release Year'] <= end)]
            tag_revenue = period_df['Tags'].str.split(',').explode().reset_index()
            tag_revenue['Tags'] = tag_revenue['Tags'].apply(normalize_tag)
            tag_revenue['Revenue'] = period_df.loc[tag_revenue['index'], 'Revenue Estimated'].values

            tag_revenue = tag_revenue.groupby('Tags')['Revenue'].sum().reset_index()
            tag_revenue.columns = ['Tags', 'Revenue']
            top_10_tags = tag_revenue.sort_values(by='Revenue', ascending=False).head(10)
            total_revenue = tag_revenue['Revenue'].sum()

            row = (i // 2) + 2
            col = (i % 2) + 1
            fig.add_trace(
                go.Pie(
                    labels=top_10_tags['Tags'],
                    values=top_10_tags['Revenue'],
                    name=f"{start}-{end}",
                    showlegend=True
                ),
                row=row, col=col
            )

            x_pos = (col - 1) * 0.5 + 0.15
            y_pos = 1 - row * 0.34
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=f"Total Revenue: ${total_revenue:,.2f}",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=12, color="black")
            )

        fig.update_layout(
            title="Revenue Analysis of Top Game <b>Tags</b> over time",
            height=1500,
            legend=dict(
                x=1,
                y=0,  
                bgcolor="rgba(255, 255, 255, 0.5)", 
                bordercolor="Black", 
                borderwidth=1  
            ),
            showlegend=True
        )

        return dcc.Graph(figure=fig, config={"responsive": True})

    @property
    def reviews_vs_revenue_over_time(self):
        """
        Creates a scatter plot showing games with revenue over 5 million USD and their reviews over time.

        Returns:
            A Dash Graph object displaying the scatter plot.
        """
        copy_df = self.df.copy()
        copy_df['Release Year'] = copy_df['Release Date'].dt.year
        copy_df.dropna(inplace=True)
        copy_df = copy_df[copy_df['Revenue Estimated'] >= 5e6]

        fig = px.scatter(
            copy_df, 
            x="Release Year", 
            y="review_score", 
            size="Revenue Estimated",
            color='review_summary', 
            hover_name="name",
            title="Games which have collected Revenue over <b>5 Million USD</b> over the years along with their <b>reviews</b>:",
            labels={"review_score": "Review Score", "Revenue Estimated": "Estimated Revenue"}
        )

        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(height=self.height, legend_title='Review Summary')

        return dcc.Graph(figure=fig)

    @property
    def production_and_revenue_over_years(self):
        """
        Creates a dual-axis plot showing game production counts and revenue earned over the years.

        Returns:
            A Dash Graph object displaying the dual-axis plot.
        """
        filtered_df = self.df.copy()
        filtered_df['Release Year'] = filtered_df['Release Date'].dt.year
        filtered_df = filtered_df[(filtered_df['Release Year'] >= 2000) & (filtered_df['Release Year'] <= 2023)]

        production_count = filtered_df.groupby('Release Year').size().reset_index(name='Game Count')
        yearly_revenue = filtered_df.groupby('Release Year')['Revenue Estimated'].sum().reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=production_count['Release Year'],
                y=production_count['Game Count'],
                name='Game Production',
                marker_color='blue'
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=yearly_revenue['Release Year'],
                y=yearly_revenue['Revenue Estimated'],
                name='Revenue Earned',
                mode='lines+markers',
                line=dict(color='red', width=2)
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title='Number of Games <b>Produced</b> and <b>Revenue</b> Over Years:',
            xaxis_title='Release Year',
            yaxis_title='Number of Games', 
            yaxis2_title='Total Revenue (Estimated)',
            legend_title='Metrics',
            height=self.height,
            template='plotly_white'
        )

        return dcc.Graph(figure=fig)

    @property
    def top_games_comparison(self):
        """
        Creates a bar chart comparing the top games by revenue and by reviews.

        Returns:
            A Dash Graph object displaying the comparison.
        """
        revenue_top = self.df.nlargest(10, 'Revenue Estimated')[['name', 'Revenue Estimated']]
        reviews_top = self.df.nlargest(10, 'Reviews Total')[['name', 'Reviews Total']]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Games by Revenue", "Top Games by Reviews"))
        fig.add_trace(
            go.Bar(x=revenue_top['name'], y=revenue_top['Revenue Estimated'], name='Revenue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=reviews_top['name'], y=reviews_top['Reviews Total'], name='Reviews'),
            row=1, col=2
        )
        fig.update_layout(
            title='Top Games by <b>Revenue</b> vs Top Games by <b>Reviews</b>',
            showlegend=False,
            height=self.height
        )
        return dcc.Graph(figure=fig)  