import pandas as pd

class DataIntegrator:
    """
    A class to integrate and merge data from two dataframes.

    Attributes:
        df1: The first dataframe to merge.
        df2: The second dataframe to merge.
    """
    def __init__(self, df1, df2):
        """
        Initializes the DataIntegrator instance with two dataframes.

        Args: 
            df1: The first dataframe.
            df2: The second dataframe.
        """
        self.df1 = df1
        self.df2 = df2

    def retrive_app_id(self, url):
        """
        Extracts the App ID from a URL.

        Args:
            url: The URL string .

        Returns:
            The App ID as an integer if extracted successfully, otherwise an empty string.
        """
        if isinstance(url, str):
            parts = url.split('/')
            if len(parts) > 4:
                try:
                    return int(parts[4])
                except ValueError:
                    return ''
        return ''

    def merge_dataframes(self):
        """
        Merges the two dataframes on the 'App ID' column after extracting it.

        Returns:
            The merged dataframe.
        """

        # Merging dataframes based on app id.
        self.df1['App ID'] = self.df1['url'].apply(self.retrive_app_id)
        merged_df = pd.merge(self.df1, self.df2, on='App ID', how='outer')
        return merged_df
