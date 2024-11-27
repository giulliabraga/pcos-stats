from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import warnings
from ydata_profiling import ProfileReport
warnings.filterwarnings("ignore")
#%matplotlib inline

class EDA():
    '''
    Customized class for generating a complete Exploratory Data Analysis (EDA) for a given dataset.
    '''

    def __init__(self, dataset, target, categorical_columns=None, numerical_columns=None, dataset_name=None):
        self.dataset = dataset
        self.target = target
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.dataset_name = dataset_name

    def quick_overview(self):
        '''
        An easier way of getting the .head() and .info()
        '''
        df = self.dataset
        print(df.info())
        display(df.head(10))
    
    def categorical_data_description(self):
        '''
        Outputs a descriptive statistics table, including the value counts and percentages for each variable.
        Displays counts in the format "value_count (percentage%)".
        Filters out variables with more than 5 unique values.
        '''
        # Identify categorical columns
        cat_cols = self.categorical_columns

        # Filter out variables with more than 5 unique values for a better visualization
        filtered_cols = [col for col in cat_cols if self.dataset[col].nunique() <= 5]

        if not filtered_cols:
            return None

        # Descriptive statistics
        stats = self.dataset[filtered_cols].describe().T

        # Combine counts and percentages into a single format
        value_counts_with_percentages = self.dataset[filtered_cols].apply(
            lambda col: col.value_counts()
            .combine(
                col.value_counts(normalize=True).mul(100).round(2),
                lambda count, pct: f"{count} ({pct}%)"
            )
        ).T

        # Output containing stats and formatted counts with percentages
        categorical_data_description = (
            pd.concat([stats, value_counts_with_percentages], axis=1)
            .fillna("-")  # Fill missing values with a dash for clarity
            .style
            .format(precision=0)
        )

        return categorical_data_description

    def get_profile_report(self):
        '''
        Outputs and saves a Pandas Profiling Report.
        '''
        dataset = self.dataset

        profile = ProfileReport(dataset,
                        title='Profile Report'
                        )    
        
        profile.to_notebook_iframe()
        
        if self.dataset_name != None:
            profile.to_file(f'../outputs/profile_report_{self.dataset_name}.html')

        return profile

    def countplots(self):
        '''
        Generate countplots for the target variable and the categorical attributes (hued by the target).
        '''
        import math

        cat_cols = self.categorical_columns
        dataset = self.dataset
        trg = self.target

        # Plotting the target variable
        fig1 = sns.countplot(x=dataset[trg])
        fig1.set_title('Target variable countplot')
        fig1.legend([-1, 1], ['No phishing', 'Phishing'])

        # Dynamically determine subplot grid size
        num_features = len(cat_cols)
        cols = 3  # Fixed number of columns
        rows = math.ceil(num_features / cols)  # Dynamically determine number of rows

        # Create subplots
        fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), dpi=200)

        # Flatten axes for iteration (handles cases with fewer variables)
        axes2 = axes2.flatten()

        # Create countplots for each categorical variable
        for ax, feature in zip(axes2, cat_cols):
            sns.countplot(data=dataset, x=feature, hue=trg, ax=ax, palette=sns.color_palette("Paired")[0:2])
            ax.set_title(f'Countplot - {feature}')

        # Hide any extra subplot axes (in case number of variables < total subplots)
        for ax in axes2[num_features:]:
            ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Save figures
        if self.dataset_name != None:
            fig1.figure.savefig(f'../outputs/target_countplot_{self.dataset_name}.png', dpi=300)
            fig2.savefig(f'../outputs/features_countplots_{self.dataset_name}.png', dpi=300)

        plt.show()

    def correlations(self):
        '''
        Generate a complete correlation matrix and one with only the correlations with module over 0.6.
        '''
        dataset = self.dataset
        target = self.target

        matrix = dataset.corr()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

        sns.heatmap(matrix, cmap="coolwarm", annot=False, ax=ax1)
        ax1.set_title('Complete correlation matrix')

        high_corr = matrix[matrix > 0.5]
        low_corr = matrix[matrix < -0.5]
        sns.heatmap(high_corr, cmap="Reds", ax=ax2)
        sns.heatmap(low_corr, cmap="Blues", ax=ax2)
        ax2.set_title('Correlações de Pearson com módulo superior a 0.5')

        plt.tight_layout()

        if self.dataset_name != None:
            plt.savefig(f'../outputs/correlations_{self.dataset_name}.png',dpi=300)

        fig.show()

    def correlation_with_target(self):
        '''
        Pearson correlation coefficient for each variable with respect to the target
        '''
        correlation_with_target = self.dataset.corr()[self.target].abs().sort_values(ascending=False)

        display(correlation_with_target)

    def complete_eda(self, categorical_columns, numerical_columns):
        '''
        Runs all the EDA functions.
        '''
        self.quick_overview()

        self.categorical_data_description()

        self.numerical_data_description()

        self.get_profile_report()

        self.countplots()

        self.correlations()