import numpy as np
import re
from tqdm import tqdm


def reformat_columns(columns_to_reformat, query_type, query_df, metric_names=[]):
    """
    Reformat the columns of the query dataframe

    Parameters:
    -----------
    columns_to_reformat : list
        List of columns to reformat
    query_type : str
        Type of query (either 'modified_cosine' or 'spectral_entropy')
    query_df : pandas.DataFrame
        Query dataframe
    
    Returns:
    --------
    query_df: pandas.DataFrame
        Query dataframe with reformatted columns
    """

    if query_type == 'spectral_entropy' and metric_names == []:
        raise ValueError('metric_names must be provided for spectral entropy queries')

    # Reformat library_spectra_matches column
    if 'library_spectra_matches' in columns_to_reformat:
        columns_to_reformat.remove('library_spectra_matches')
        query_df['library_spectra_matches'] = reformat_library_matches(query_df, 'library_spectra_matches')

    # Reformat other columns
    for column in columns_to_reformat:
        if column in metric_names:
            query_df[column] = reformat_library_matches(query_df, column)
        else:
            query_df[column] = reformat_other_columns(query_df, column)

    return query_df


def reformat_library_matches(query_df, column):
    """
    Reformat the library_spectra_matches column of the query dataframe

    Parameters:
    -----------
    query_df : pandas.DataFrame
        Query dataframe
    column : str
        Column name

    Returns:
    --------
    new_column: pandas.DataFrame column
        reformatted library_spectra_matches column
    """

    # Reformat column
    all_matches = {}
    for query in tqdm(query_df.index.values):
        matches = []
        non_decimal = re.compile(r'[^\d.]+')

        test = query_df.loc[query][column].replace("'", "").replace(')', '').split('(')[1:]
        for pair in test:
            matches.append((pair.split(',')[0], float(non_decimal.sub('', pair.split(',')[1]))))

        all_matches[query] = matches
    new_column = query_df.index.map(all_matches)

    return new_column


def reformat_other_columns(query_df, column_name):
    """
    Reformat exact_matches column of the query dataframe

    Parameters:
    -----------
    query_df : pandas.DataFrame
        Query dataframe

    Returns:
    --------
    new_column: pandas.DataFrame column
       reformatted exact_matches column
    """

    # Reformat column
    column_vals = {}
    for query in tqdm(query_df.index.values):
        if 'exact_matches' in column_name:
            column_vals[query] = np.array(query_df.loc[query, column_name].replace('[', '') \
                                          .replace(']', '').replace('\'', '') \
                                          .replace(',', '').split()).astype(int)
        elif 'tanimoto' in column_name:
            column_vals[query] = np.array(query_df.loc[query, column_name].replace('[', '') \
                                          .replace(']', '').replace('\'', '') \
                                          .replace(',', '').split()).astype(float)
        else:
            column_vals[query] = np.array(query_df.loc[query, column_name].replace('[', '') \
                                          .replace(']', '').replace('\'', '') \
                                          .replace(',', '').split())
    new_column = query_df.index.map(column_vals)

    return new_column
