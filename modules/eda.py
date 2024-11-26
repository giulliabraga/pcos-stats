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

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def quick_overview(self):
        '''
        An easier way of getting the .head() and .info()
        '''
        df = self.dataset
        print(df.info())
        display(df.head(10))

    def categorical_data_description(self, categorical_columns):
        '''
        Outputs a descriptive statistics table, including the value counts for each variable.
        Only works well with datasets in which the features have a limitted amount of unique values.
        '''
        # Descriptive statistics
        stats = self.dataset.describe().T

        # Unique value counts
        value_counts = self.dataset.apply(lambda col: col.value_counts()).T

        # Output containin the stats and the counts for each unique value
        categorical_data_description = (
                                            pd.concat([stats,value_counts],axis=1)
                                            .fillna(0)
                                            .style
                                            .format(precision=0)
                                        )
        
        display(categorical_data_description)

        return categorical_data_description   

    def get_profile_report(self):
        '''
        Outputs and saves a Pandas Profiling Report.
        '''
        dataset = self.dataset

        profile = ProfileReport(dataset,
                        title='Phishing Websites Dataset'
                        )    
        
        profile.to_notebook_iframe()
        
        profile.to_file('../outputs/profile_phishing_websites.html')

        return profile

    def countplots(self, categorical_columns):
        '''
        Generate countplots for the target variable and the categorical attributes (hued by the target).
        '''
        dataset = self.dataset
        trg = self.target

        # Plotting the target variable
        fig1 = sns.countplot(x=dataset[trg])
        fig1.set_title('Target variable countplot')
        fig1.legend([-1,1],['No phishing', 'Phishing'])

        fig2, axes2 = plt.subplots(10, 3, figsize=(20, 30), dpi=200)
        for ax, feature in zip(axes2.flat, categorical_columns):
            sns.countplot(data=dataset, x=feature, hue=trg, ax=ax, palette=sns.color_palette("Paired")[0:2])
            ax.set_title(f'Countplot - {feature}')
        
        plt.tight_layout()

        fig1.figure.savefig('../outputs/target_countplot.png', dpi=300)
        fig2.savefig('../outputs/features_countplots.png', dpi=300)

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

        high_corr = matrix[matrix > 0.6]
        low_corr = matrix[matrix < -0.6]
        sns.heatmap(high_corr, cmap="Reds", ax=ax2)
        sns.heatmap(low_corr, cmap="Blues", ax=ax2)
        ax2.set_title('Correlations with module over 0.6')

        plt.tight_layout()
        plt.savefig('../outputs/correlations.png',dpi=300)
        fig.show()

    def complete_eda(self, categorical_columns, numerical_columns):
        '''
        Runs all the EDA functions.
        '''
        self.quick_overview()

        self.categorical_data_description(categorical_columns=categorical_columns)

        self.numerical_data_description(numerical_columns=numerical_columns)

        self.get_profile_report()

        self.countplots(categorical_columns=categorical_columns)

        self.correlations()