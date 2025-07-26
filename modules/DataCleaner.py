from modules.DataIntegrator import DataIntegrator

from datetime import datetime
import pandas as pd
import numpy as np
import re

class DataCleaner:
    """
    A class to clean and process data for Steam game analysis.

    Attributes:
        df1: The first dataframe for integration.
        df2: The second dataframe for integration.
        integrator: An instance of DataIntegrator for merging data.
        df: The resulting dataframe after integration.
    """

    def __init__(self, df1, df2):
        """
        Initializes the DataCleaner instance with two dataframes and merges them.

        Args:
            df1: The first dataframe.
            df2: The second dataframe.
        """
        self.df1 = df1
        self.df2 = df2
        self.integrator = DataIntegrator(self.df1, self.df2)
        self.df = self.integrator.merge_dataframes()

    def fill_game_names(self):
        """
        Fills missing game names by parsing them from the URL column.

        Returns:
            The updated dataframe with game names filled.
        """
        def parse_game_name(url):
            if isinstance(url, str):
                return url.split('/')[-2]
            return np.nan

        self.df['name'] = self.df.apply(lambda row: parse_game_name(row['url']) if pd.isnull(row['name']) else row['name'], axis=1)
        self.df['name'] = self.df['name'].apply(lambda x: x.replace("_", " ") if isinstance(x, str) else x)
        self.df = self.df.drop('url', axis=1)
        return self.df

    def drop_unnecessary_columns(self):
        """
        Drops columns that are unnecessary for the analysis.

        Returns:
            The updated dataframe with unnecessary columns removed.
        """
        columns_to_drop = [
            'types', 'discount_price', 'developer', 'recent_reviews', 'Reviews D7', 'Reviews D30', 'Reviews D90', 'name_slug',
            'desc_snippet', 'mature_content', 'achievements', 'publisher', 'Modified Tags', 'App ID', 'Steam Page',
            'minimum_requirements', 'recommended_requirements', 'languages', 'genre',
            'game_description', 'discount_price', 'game_details'
            ]
        self.df = self.df.loc[:, ~self.df.columns.isna()]
        self.df = self.df.drop(columns=columns_to_drop, axis=1)
        return self.df

    def format_date(self, date_col):
        """
        Formats date strings in a specified column to datetime objects.
 
        Args:
            date_col: The column name containing date strings.

        Returns:
            The updated dataframe with formatted dates.
        """
        def parse_date(date_str):
            if pd.isnull(date_str):
                return np.nan
            try:
                return datetime.strptime(date_str.strip(), "%b %d, %Y").date()
            except ValueError:
                return np.nan

        self.df[date_col] = self.df[date_col].apply(parse_date)
        return self.df

    def fill_missing_values(self, missing_columns):
        """
        Fills missing values in specified columns with an empty string.

        Args:
            missing_columns: List of column names to fill missing values.

        Returns:
            The updated dataframe with missing values filled.
        """
        self.df[missing_columns] = self.df[missing_columns].fillna('')
        return self.df

    def integrate_columns(self):
        """
        Integrates tags from multiple columns into a single column.

        Returns:
            The updated dataframe with integrated tags.
        """
        def join_tags(row):
            tags1 = row['popular_tags'].split(',')
            tags2 = row['Tags'].split(',')

            tags1 = [tag.strip() for tag in tags1 if tag.strip()]
            tags2 = [tag.strip() for tag in tags2 if tag.strip()]

            return ','.join(list(set(tags2 + tags1)))

        self.df['Tags'] = self.df.apply(join_tags, axis=1)
        self.df = self.df.drop('popular_tags', axis=1)
        return self.df

    def combine_cols(self, col1, col2):
        """
        Combines two columns, preferring the first, and drops the second.

        Args:
            col1: The primary column to keep.
            col2: The secondary column to combine and drop.

        Returns:
            The updated dataframe with combined columns.
        """
        self.df[col1] = self.df[col1].combine_first(self.df[col2])
        self.df = self.df.drop(col2, axis=1)
        return self.df

    def convert_reviews_to_float(self):
        """
        Converts review scores to floats.

        Returns:
            The updated dataframe with review scores as floats.
        """
        def parse_reviews(review_score):
            try:
                return float(review_score.replace('%', '').replace(',', '.'))
            except AttributeError:
                return np.nan

        self.df['review_score'] = self.df['Reviews Score Fancy'].apply(parse_reviews)
        self.df = self.df.drop('Reviews Score Fancy', axis=1)
        return self.df

    def convert_price_to_float(self):
        """
        Converts price strings to float values.

        Returns:
            The updated dataframe with prices as floats.
        """
        self.df['launch_price'] = (
            self.df['Launch Price']
            .str.replace('$', '', regex=False)
            .str.replace(',', '.', regex=False)
        )

        self.df['launch_price'] = pd.to_numeric(self.df['launch_price'], errors='coerce').fillna(0)
        self.df = self.df.drop('Launch Price', axis=1)
        return self.df

    def drop_null_values(self):
        """
        Drops rows with null values in critical columns.

        Returns:
            The updated dataframe with null values removed.
        """
        self.df = self.df.dropna(subset=['Release Date'])
        return self.df

    def convert_revenue_to_float(self):
        """
        Converts revenue strings to float values.

        Returns:
            The updated dataframe with revenue as floats.
        """
        def parse_revenue(rev_str):
            if isinstance(rev_str, str):
                rev_str = re.sub(r'[^\d,]', '', rev_str).replace(',', '.')
                return float(rev_str)
            else:
                return np.nan

        self.df['Revenue Estimated'] = self.df['Revenue Estimated'].apply(parse_revenue)
        return self.df

    def add_review_summary(self):
        """
        Adds a review summary column from the 'all_reviews' column.

        Returns:
            The updated dataframe with the review summary column.
        """
        self.df['review_summary'] = self.df['all_reviews'].str.split(',').str[0]
        self.df.replace(r'^\d+ user reviews$', '', regex=True, inplace=True)
        return self.df

    def fill_null_of_all_reviews(self):
        """
        Fills null values in review score and total reviews by extracting data from 'all_reviews'.

        Returns:
            The updated dataframe with missing review data filled.
        """
        def extract_data(row):
            count_match = re.search(r"\(([\d,]+)\)", row)
            score_match = re.search(r"(\d+)%", row)

            count = int(count_match.group(1).replace(",", "")) if count_match else np.nan
            score = int(score_match.group(1)) if score_match else np.nan

            return count, score

        for index, row in self.df.iterrows():
            if pd.isna(self.df.at[index, "Reviews Total"]) or pd.isna(self.df.at[index, "review_score"]):
                count, score = extract_data(row["all_reviews"])
                if pd.isna(self.df.at[index, "Reviews Total"]):
                    self.df.at[index, "Reviews Total"] = count
                if pd.isna(self.df.at[index, "review_score"]):
                    self.df.at[index, "review_score"] = score

        self.df = self.df.drop('all_reviews', axis=1)
        return self.df

    def fill_review_score_reviews_total(self):
        """
        Fills missing review scores and totals with statistical values.

        Returns:
            The updated dataframe with filled review data.
        """
        self.df = self.df[(self.df['Reviews Total'] != 0) & (self.df['review_score'] != 0)]

        self.df["Reviews Total"] = self.df["Reviews Total"].fillna(self.df["Reviews Total"].median()) # Data was skewed for this feature
        self.df["review_score"] = self.df["review_score"].fillna(self.df["review_score"].mean())
        return self.df

    def clean_data(self):
        """
        Executes a sequence of cleaning actions on the dataframe.

        Returns:
            The final fully cleaned dataframe.
        """

        # Actions to done one by one (cleaning)
        actions = [
            lambda: self.drop_unnecessary_columns(),
            lambda: self.fill_missing_values(missing_columns=['all_reviews', 'release_date', 'popular_tags', 'Tags']),
            lambda: self.format_date('release_date'),
            lambda: self.fill_game_names(),
            lambda: self.integrate_columns(),
            lambda: self.combine_cols('Release Date', 'release_date'),
            lambda: self.combine_cols('name', 'Title'),
            lambda: self.combine_cols('Launch Price', 'original_price'),
            lambda: self.convert_price_to_float(),
            lambda: self.convert_reviews_to_float(),
            lambda: self.drop_null_values(),
            lambda: self.convert_revenue_to_float(),
            lambda: self.add_review_summary(),
            lambda: self.fill_null_of_all_reviews(),
            lambda: self.fill_review_score_reviews_total(),
        ]

        for action in actions:
            self.df = action()
        return self.df
