# Code for custom code recipe compute_RECOMMENDATION (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import *

# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
#input_A_names = 
# The dataset objects themselves can then be created like this:
#input_A_datasets = [dataiku.Dataset(name) for name in input_A_names]

# For outputs, the process is the same:
#output_A_names = 

# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
#my_variable = 

# For optional parameters, you should provide a default value in case the parameter is not present:
#my_variable = get_recipe_config().get('parameter_name', None)

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from numpy.linalg import norm

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
e_commerce = dataiku.Dataset(get_input_names_for_role('ecommercedata')[0])
df = e_commerce.get_dataframe()
max_features= int(get_recipe_config()['max_features'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df.columns = ['PID','DESC','CID'] # TODO: Not to include in final
#df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
desc_by_pid = df.groupby(['PID','DESC']).count().where(lambda x: x.CID > 0).dropna().reset_index()[['PID','DESC']]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
desc_by_pid.DESC = desc_by_pid.DESC.apply(lambda x: x.lower()).apply(lambda y: ' '.join(list(set(y.split()) - stop_words)))#.iloc[4]#.strip(stop_words)#stop_words

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pid_by_desc_dict = desc_by_pid.set_index('DESC').to_dict(orient='index')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Maps a product description to a vector
vectorizer = TfidfVectorizer(stop_words = stop_words, lowercase=True, analyzer ='word', max_features=max_features)
X = vectorizer.fit_transform(desc_by_pid.DESC)
X = X.todense()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Constructs a similarity matrix between pairs of products
n_lines = X.shape[0]
M = np.zeros( (n_lines,n_lines))
for i in tqdm(range(X.shape[0])):
    for j in range(X.shape[0]):
        if i > j:
            value = X[i].dot(X[j].transpose())/(norm(X[i])*norm(X[j]))
            if value is np.nan:
                value = 0
            M[i,j] = value
            M[j,i] = value

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Search the purchases made by user
df_by_CID = df.set_index('CID').copy()
cid_by_user = {}
for userid in tqdm(df.CID.unique()):
    try:
        cid_by_user.update(df_by_CID.loc[[userid]].reset_index().groupby('CID')['PID'].agg(list).to_dict())
    except KeyError:
        print('KeyError')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Search the most similar product for each product bought in the past and eliminate duplicates.
pid_to_midx = desc_by_pid['PID'].reset_index().set_index('PID').to_dict()['index']
midx_to_pid = desc_by_pid['PID'].to_dict()
recos_per_user = {}
for user in tqdm(cid_by_user.keys()):
    recos_list=[]
    for item in cid_by_user[user]:
        recos_list.append(midx_to_pid[np.nan_to_num(M[pid_to_midx[item]]).argmax()])
    recos_per_user[user] = list(set(recos_list) - set(cid_by_user[user]))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# make a dataframe
desc_by_pid_dict = desc_by_pid.set_index('PID')['DESC'].to_dict()
pds = []
for user in tqdm(recos_per_user.keys()):
    if len(recos_per_user[user]) > 0:
        s0 = pd.Series(data = [user for i in range(len(recos_per_user[user]))])
        s1 = pd.Series(data = recos_per_user[user])
        s2 = pd.Series(data = [desc_by_pid_dict[recos_per_user[user][i]] for i in range(len(recos_per_user[user]))])
        pds.append(pd.DataFrame([s0,s1,s2]).transpose().set_index(0)) 

        
recommendation_df = pd.concat(pds)

recommendation_df = recommendation_df.reset_index()
recommendation_df.columns = ['UID','PID','DESC'] #
recommendation_df.UID = recommendation_df.UID.astype(int)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

# Write recipe outputs
recommendation = dataiku.Dataset(get_output_names_for_role("RECOMMENDATION")[0])
recommendation.write_with_schema(recommendation_df)