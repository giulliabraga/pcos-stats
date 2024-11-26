import numpy as np
import pandas as pd
from scipy.stats import kstest, chi2_contingency
from IPython.display import display

class DatasetIntegration():

    def __init__(self, dataset_1, dataset_2, numerical_columns, categorical_columns):
        self.ds1 = dataset_1
        self.ds2 = dataset_2
        self.num_cols = numerical_columns
        self.cat_cols = categorical_columns

    def run_ks_test(self):
            '''
            Run the Komolgorov-Smirnov test of goodness-of-fit to compare the distributions of each numerical variable in two datasets. 
            We are using the two-sided test, so the null hypothesis is that the two distributions are identical, F(x)=G(x) for all x; the alternative is that they are not identical.
            Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
            '''
            list_of_statistics = []
            list_of_pvalues = []
            list_of_diff_flags = []

            for num_col in self.num_cols:

                statistic, p_value = kstest(self.ds1[num_col], self.ds2[num_col], alternative='two-sided')

                if p_value < 0.05:
                    are_different = True
                else:
                    are_different = False

                '''
                print(f'Feature: {metric}')
                print(f'Statistic: {statistic}')
                print(f'p-value: {p_value}')
                print(f'It is {are_different} that the classifiers performance for this metric is signifficantly different.')
                '''

                list_of_statistics.append(statistic)
                list_of_pvalues.append(p_value)
                list_of_diff_flags.append(are_different)


            ks_dict = {
                'Variável': self.num_cols,
                'Estatísica': list_of_statistics,
                'p-valor': list_of_pvalues,
                'São diferentes': list_of_diff_flags
            }

            ks_results = pd.DataFrame(ks_dict)

            num_cols_with_no_difference = np.array(ks_results[ks_results['São diferentes'] == False]['Variável'].tolist())

            return ks_results, num_cols_with_no_difference

    def run_chi2_test(self):
        '''
        Run the Chi2 contingency test to evaluate the independence of categorical variables accross two datasets.
        '''
        list_of_statistics = []
        list_of_pvalues = []
        list_of_diff_flags = []

        for cat_col in self.cat_cols:
            contingency_table = pd.crosstab(self.ds1[cat_col], self.ds2[cat_col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

            are_different = p_value < 0.05

            list_of_statistics.append(chi2_stat)
            list_of_pvalues.append(p_value)
            list_of_diff_flags.append(are_different)

        chi2_dict = {
            'Variável': self.cat_cols,
            'Estatística': list_of_statistics,
            'p-valor': list_of_pvalues,
            'São diferentes': list_of_diff_flags
        }

        chi2_results = pd.DataFrame(chi2_dict)

        cat_cols_with_no_difference = np.array(chi2_results[chi2_results['São diferentes'] == False]['Variável'].tolist())

        return chi2_results, cat_cols_with_no_difference

    def merge_datasets(self):
        '''
        Merge two datasets by selecting only the columns for which the distributions were considered identical according to statistical tests.
        '''
        ks_results , num_cols_to_merge = self.run_ks_test()
        chi2_results , cat_cols_to_merge = self.run_chi2_test()
        print(f'Resultados do teste Komolgorov-Smirnov: \n')
        display(ks_results)
        print(f'\nResultados do teste Chi-Quadrado: \n')
        display(chi2_results)

        selected_variables = np.concatenate((num_cols_to_merge, cat_cols_to_merge), axis=0)

        merged_datasets = pd.concat([self.ds1[selected_variables], self.ds2[selected_variables]])

        merged_datasets = merged_datasets.sample(frac=1).reset_index(drop=True)

        return merged_datasets