import os
import pathlib

# We have already done hyperparameter optimization and the below values result in the best score.
batch_size_list = [8]  # Batch size: [8,16,32]
num_epochs_list = [3]
token_list = ["special"]  # the token_list for the bert model can have those values =['special', '[SEP]',' '].
numerical_features_list = [
    ["nbr_PREMISE_Fact", "nbr_PREMISE_Hypothesis", "nbr_PREMISE_Other", "nbr_PREMISE_RealExample",
     "nbr_PREMISE_Statistic"]]
textual_features_lists = [
    ["claim_text", "premise_texts"]]  # [['claim_label','claim_text','relation_types','premise_labels','premise_texts']]
lr_list = [2e-5]  # Learning rate (Adam): [5e-5,3e-5,2e-5 ]

seed = 42
test_size = 0.20
target_class = "STRONG"  # PERSUASIVE, STRONG, SPECIFIC, OBJECTIVE, TEMPORALHISTORY
loss_function_name = "CELoss"
task = "classification"
cross_validation = 10

# Path
root = ''
for root_relative_dir in [r'./', r'../', r'../../', r'../../../']:
    if all(pathlib.Path(os.path.abspath(root_relative_dir + subdir)).exists() for subdir in ['models/', 'docs/']):
        root = os.path.abspath(root_relative_dir)
        break
else:
    raise Exception('Unable to find root directory. Working directory is', os.path.abspath('./'))
df_result_csv_path = f"{root}/results/df_result.csv"
dataset_csv_path = f"{root}/Data_processing/earningsCall_arg_quality.csv"
