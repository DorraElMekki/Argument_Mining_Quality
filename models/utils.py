import logging
import os
import stat
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_metric
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AdamW, AutoTokenizer, get_scheduler

from models.constants import seed, cross_validation
from models.custom_sequence_classification import CustomSequenceClassification

metrics_dict = {
    "metric_accuracy": load_metric("accuracy"),
    "metric_matthewscorrelation": load_metric("matthews_correlation"),
    "metric_pearsonr": load_metric("pearsonr"),
    "metric_precision_none": load_metric("precision"),
    "metric_precision_micro": load_metric("precision"),
    "metric_precision_macro": load_metric("precision"),
    "metric_precision_weighted": load_metric("precision"),
    "metric_recall_none": load_metric("recall"),
    "metric_recall_micro": load_metric("recall"),
    "metric_recall_macro": load_metric("recall"),
    "metric_recall_weighted": load_metric("recall"),
    "metric_spearmanr": load_metric("spearmanr"),
    "metric_f1_none": load_metric("f1"),
    "metric_f1_micro": load_metric("f1"),
    "metric_f1_macro": load_metric("f1"),
    "metric_f1_weighted": load_metric("f1"),
}


def save_metrics(df_result: pd.DataFrame, dataset_name: str = "train",
                 metrics_dict=metrics_dict) -> pd.DataFrame:
    """
    save the scores of the metrics defined in metrics_dict in the df_result.
    :param df_result:
    :param dataset_name: can take the values: train or test
    :param metrics_dict:
    :return: we return df_result with the scores
    """
    if metrics_dict is None:
        metrics_dict = metrics_dict
    for metric_name, metric in metrics_dict.items():
        if len(metric_name.split("_")) == 3:
            if metric_name.split("_")[2] == "none":
                metric_value = metric.compute(average=None)
            else:
                metric_value = metric.compute(average=metric_name.split("_")[2])
        else:
            metric_value = metric.compute()
        if metric_name == "metric_matthewscorrelation":
            df_result[metric_name + "_" + dataset_name] = metric_value["matthews_correlation"]
        elif isinstance(metric_value[metric_name.split("_")[1]], float):
            df_result[metric_name + "_" + dataset_name] = metric_value[metric_name.split("_")[1]]  # one float
        else:
            df_result[metric_name + "_" + dataset_name] = [metric_value[metric_name.split("_")[1]]]  # list of values
        logging.info(metric_value)
    return df_result


def create_folder_if_does_not_exist(folder_path: str) -> None:
    """
    create folder if not exist

    :param folder_path:
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def process_data_to_DatasetDict_type(dataset_csv_path: str, target_class: str, fold_id: int, token: str,
                                     textual_features_list:
                                     List[str], numerical_features: List[str]) -> Tuple[DatasetDict, int]:
    """
    process data to be used as input to the model
    the dataset type = DatasetDict

    :param dataset_csv_path:
    :param target_class:
    :param fold_id:
    :param token:
    :param textual_features_list:
    :param numerical_features:
    :return:
    """
    df = pd.read_csv(dataset_csv_path)
    if token == "special":

        def combined_textual_features_special_token(row: pd.Series, textual_features_list: List[str]) -> str:
            """

            :param row:
            :param textual_features_list:
            :return: return the different textual features concatenated using the special tokens such as [cl_label] instead of [SEP].
            """
            combined = ""
            if "claim_label" in textual_features_list:
                combined += " [cl_label] " + (str(row["claim_label"])) + " [/cl_label] "
            if "claim_text" in textual_features_list:
                combined += " [cl_text] " + (str(row["claim_text"])) + " [/cl_text] "
            for i in range(len(eval(row["relation_ids"]))):
                if "relation_ids" in textual_features_list:
                    combined += " [R_id] " + (str(eval(row["relation_ids"])[i])) + " [/R_id] "
                if "relation_types" in textual_features_list:
                    combined += " [R_type] " + (str(eval(row["relation_types"])[i])) + " [/R_type] "
                if "premise_labels" in textual_features_list:
                    combined += " [pr_label] " + (str(eval(row["premise_labels"])[i])) + " [/pr_label] "
                if "premise_texts" in textual_features_list:
                    combined += " [pr_text] " + (str(eval(row["premise_texts"])[i])) + " [/pr_text] "
            return combined

        df["combined_textual_features"] = df.apply(
            lambda row: combined_textual_features_special_token(row, textual_features_list), axis=1)
    elif token in {"[SEP]", " "}:

        def combined_textual_features(row: pd.Series, textual_features_list: List[str], token: str) -> str:
            """

            :param row:
            :param textual_features_list:
            :param token:
            :return: return the different textual features concatenated using the space or [SEP] token.
            """
            combined: str = ""
            if token == "[SEP]":
                token = " [SEP] "
            if "claim_label" in textual_features_list:
                combined += (str(row["claim_label"])) + token
            if "claim_text" in textual_features_list:
                combined += (str(row["claim_text"])) + token
            for i in range(len(eval(row["relation_ids"]))):
                if "relation_ids" in textual_features_list:
                    combined += (str(row["relation_ids"])) + token
                if "relation_types" in textual_features_list:
                    combined += (str(eval(row["relation_types"])[i])) + token
                if "premise_labels" in textual_features_list:
                    combined += (str(eval(row["premise_labels"])[i])) + token
                if "premise_texts" in textual_features_list:
                    combined += (str(eval(row["premise_texts"])[i])) + token
            return combined

        df["combined_textual_features"] = df.apply(
            lambda row: combined_textual_features(row, textual_features_list, token), axis=1)

    logging.info(f"df.head(2) {df.head(2)}")
    possible_labels = df[target_class].unique()
    possible_labels.sort()
    num_labels = len(possible_labels)
    logging.info(f"possible_labels for {target_class}:{possible_labels}")

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df["label"] = df[target_class].replace(label_dict)
    logging.info(f"label_dict for {target_class}:{label_dict}")
    possible_labels = df["claim_label"].unique()

    claim_label_dict = {}
    for index, possible_claim_label in enumerate(possible_labels):
        claim_label_dict[possible_claim_label] = index
    df["claim_label"] = df["claim_label"].replace(claim_label_dict)
    logging.info(f"claim_label_dict for claim_label:{claim_label_dict}")

    def add_extra_data(row: pd.Series, numerical_features: List[str]) -> List[str]:
        """

        :param row:
        :param numerical_features:
        :return: return a list of the extra categorical features that we would like to include in our model.
        """
        extra_data = []
        for numerical_feature in numerical_features:
            extra_data.append(row[numerical_feature])
        return extra_data

    df["extra_data"] = df.apply(lambda row: add_extra_data(row, numerical_features), axis=1)

    def k_fold_classification(df: pd.DataFrame, cross_validation: int) -> pd.Series:

        """
         stratified k fold split
        :param df: all columns will be used for the stratified split
        :param cross_validation: number of folds
        :return:
        """
        assert df.isna().sum().sum() == 0

        _df = df.astype("category")
        _df = _df.apply(lambda x: x.cat.codes)  # convert into numerical

        fold = pd.Series(index=df.index, dtype="int32")
        stratifier = IterativeStratification(n_splits=cross_validation, order=2)  #
        X = np.zeros(len(df))

        stratifier_generator = stratifier.split(X, y=_df.values)
        for i, (_, y_idx) in enumerate(stratifier_generator):
            fold[y_idx] = i
        return fold

    columns = [
        "ann_id",
        "ann_file_id",
        "claim_label",
        "STRONG",
        "nbr_ATTACK",
        "nbr_SUPPORT",
        "nbr_PREMISE_Fact",
        "nbr_PREMISE_Hypothesis",
        "nbr_PREMISE_Other",
        "nbr_PREMISE_RealExample",
        "nbr_PREMISE_Statistic",
        "nbr_tokens_per_premises_list",
        "nbr_sentences_per_premises_list",
        "nbr_sentences_per_claim",
        "nbr_tokens_per_claim",
        "year",
        "quarter",
        "nbr_premises",
        "company_name",
        "nbr_sentences_per_argument",
        "nbr_tokens_per_argument",
        "sum_nbr_tokens_per_premises_list",
        "sum_nbr_sentences_per_premises_list",
    ]
    df["fold_id"] = k_fold_classification(df[columns], cross_validation)

    def train_test_split_fold_id(df: pd.DataFrame, fold_id: int) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        split the train, test sets with respect to the fold_id.
        :param df:
        :param fold_id:
        :return:
        """
        X_train = df[df["fold_id"] != fold_id]["combined_textual_features"].index.values
        X_val = df[df["fold_id"] == fold_id]["combined_textual_features"].index.values
        y_train = df[df["fold_id"] != fold_id].label.values
        y_val = df[df["fold_id"] == fold_id].label.values
        return X_train, X_val, y_train, y_val

    X_train, X_val, y_train, y_val = train_test_split_fold_id(df, fold_id)

    # add a column for data_type in the df
    df["data_type"] = ["not_set"] * df.shape[0]
    df.loc[X_train, "data_type"] = "train"
    df.loc[X_val, "data_type"] = "test"

    # create the dataset in the type DatasetDict

    dataset = {
        "train": Dataset.from_dict(
            {
                "label": df[df.data_type == "train"].label.values,
                "extra_data": df[df.data_type == "train"].extra_data.values,
                "text": df[df.data_type == "train"].combined_textual_features.values,
            }
        ),
        "test": Dataset.from_dict(
            {
                "label": df[df.data_type == "test"].label.values,
                "extra_data": df[df.data_type == "test"].extra_data.values,
                "text": df[df.data_type == "test"].combined_textual_features.values,
            }
        ),
    }
    dataset = DatasetDict(dataset)
    logging.info(f"dataset:{dataset}")
    return dataset, num_labels


def data_tokenization(dataset: DatasetDict, model_name: str, batch_size: int) -> Tuple[
    AutoTokenizer, DataLoader, DataLoader]:
    """

    :param dataset:
    :param model_name:
    :param batch_size:
    :return:
    """
    logging.info(f"example from the dataset:{dataset['train'][0]}")
    # Prepare a dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Train in native PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    logging.info(f"tokenized_datasets:{tokenized_datasets}")
    tokenized_datasets.set_format("torch")
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=seed)
    ##DataLoader
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)
    return tokenizer, train_dataloader, eval_dataloader


def train_model(model_name: str, tokenizer: AutoTokenizer, num_labels: int, token: str, lr: float, num_epochs: int,
                train_dataloader: DataLoader, device: str, weights: List[int],
                metrics_dict: dict, df_result: pd.DataFrame, num_extra_dims: int) -> Tuple[
    CustomSequenceClassification, pd.DataFrame]:
    """

    :param model_name:
    :param tokenizer:
    :param num_labels:
    :param token:
    :param lr:
    :param num_epochs:
    :param train_dataloader:
    :param device:
    :param weights:
    :param metrics_dict:
    :param df_result:
    :param num_extra_dims: the size of numerical features
    :return:
    """
    model = CustomSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                         num_extra_dims=num_extra_dims)  # num_extra_dims the size of numerical features
    if token == "special":
        list_of_added_tokens = [
            "[cl_text]",
            "[/cl_text]",
            "[cl_label]",
            "[/cl_label]",
            "[R_id]",
            "[/R_id]",
            "[R_type]",
            "[/R_type]",
            "[pr_label]",
            "[/pr_label]",
            "[pr_text]",
            "[/pr_text]",
            "[nbr_support]",
            "[/nbr_support]",
        ]
        num_added_toks = tokenizer.add_tokens(list_of_added_tokens, special_tokens=True)
        print("num_added_toks=", num_added_toks)
        model.resize_token_embeddings(len(tokenizer))

    ##Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    model.to(device)

    ##Training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    loss_train_total = 0
    loss_epochs = []
    logits_epochs = []
    predictions_epochs = []
    for epoch in range(num_epochs):
        logits_tensor_train = torch.tensor(()).to(device)
        prediction_tensor_train = torch.tensor(()).to(device)
        True_labels_tensor_train = torch.tensor(()).to(device)
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            logits_tensor_train = torch.cat((logits_tensor_train, logits.float()), 0)
            prediction_tensor_train = torch.cat((prediction_tensor_train, predictions.float()), 0)
            True_labels_tensor_train = torch.cat((True_labels_tensor_train, batch["labels"].float()), 0)

            # loss function
            class_weights = torch.FloatTensor(weights).cuda()  # add weight due to the Imbalanced data
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # train evaluation
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            for metric_name, metric in metrics_dict.items():
                metric.add_batch(predictions=predictions, references=batch["labels"])
            loss_train_total += loss.item()
        loss_epochs.append(loss_train_total)
        logits_epochs.append(logits_tensor_train.tolist())
        predictions_epochs.append(prediction_tensor_train.tolist())
    loss_avg_train = loss_train_total / len(train_dataloader)
    df_result = save_metrics(df_result, "train", metrics_dict)
    df_result["loss_avg_train"] = loss_avg_train

    return model, df_result


def model_evaluation(device: str, model: CustomSequenceClassification, weights: List[int], metrics_dict: dict,
                     df_result: pd.DataFrame, eval_dataloader: DataLoader) -> pd.DataFrame:
    """

    :param device:
    :param model:
    :param weights:
    :param metrics_dict:
    :param df_result:
    :param eval_dataloader:
    :return:
    """
    loss_test_total = 0
    logits_tensor_test = torch.tensor(()).to(device)
    prediction_tensor_test = torch.tensor(()).to(device)
    True_labels_tensor_test = torch.tensor(()).to(device)
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # save intermediate results
        logits_tensor_test = torch.cat((logits_tensor_test, logits.float()), 0)
        prediction_tensor_test = torch.cat((prediction_tensor_test, predictions.float()), 0)
        True_labels_tensor_test = torch.cat((True_labels_tensor_test, batch["labels"].float()), 0)
        # loss function
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(logits, batch["labels"])
        loss_test_total += loss.item()
        for metric in metrics_dict.values():
            metric.add_batch(predictions=predictions, references=batch["labels"])
    df_result = save_metrics(df_result, "test", metrics_dict)
    return df_result


def save_model_description_in_df_results(df_result: pd.DataFrame, model_name: str, target_class: str,
                                         weights: List[int], num_epochs: int, loss_function_name: str, task: str,
                                         fold_id: int, cross_validation,
                                         textual_features_list: List[str], numerical_features: List[str], lr: float,
                                         batch_size: int, token: str, dataset: DatasetDict) -> pd.DataFrame:
    """save general model description
    :param dataset:
    :param token:
    :param df_result:
    :param model_name:
    :param target_class:
    :param weights:
    :param num_epochs:
    :param loss_function_name:
    :param task:
    :param fold_id:
    :param cross_validation:
    :param textual_features_list:
    :param numerical_features:
    :param lr:
    :param batch_size:
    :return:
    """
    df_result["model"] = model_name
    df_result["target_class"] = target_class
    df_result["weights"] = str(weights)
    df_result["num_epochs"] = num_epochs
    df_result["support_train"] = len(dataset["train"])
    df_result["support_test"] = len(dataset["test"])
    df_result["loss_function_name"] = loss_function_name
    df_result["task"] = task
    df_result["textual_features_list"] = str(textual_features_list)
    df_result["numerical_features"] = str(numerical_features)
    df_result["token"] = token
    df_result["fold_id"] = fold_id
    df_result["cross_validation"] = cross_validation
    df_result["lr"] = lr
    df_result["batch_size"] = batch_size
    return df_result


def save_results_to_csv(df_result_csv_path: str, df_result: pd.DataFrame) -> None:
    """
    save results in csv file
    :param df_result_csv_path:
    :param df_result:
    """
    if os.path.exists(df_result_csv_path):
        logging.info(f"{df_result_csv_path} file exists")
        df = pd.read_csv(df_result_csv_path)
        new_df = pd.concat([df, df_result], ignore_index=True)
        # Drop first column of dataframe
        new_df = new_df.iloc[:, 1:]
        new_df.to_csv(df_result_csv_path)
    else:
        logging.info(f"{df_result_csv_path} file does not exist")
        df_result.to_csv(df_result_csv_path)
    logging.info(df_result)


def add_mean_std_sem_for_the_previous_scores_resulting_from_cross_validation(cross_validation: int,
                                                                             df_result_csv_path: str) -> None:
    """
    calculate the mean, standard deviation, standard error of the mean
    :param cross_validation:
    :param df_result_csv_path:
    """
    df = pd.read_csv(df_result_csv_path)

    serie_mean, serie_std, serie_sem = pd.Series(), pd.Series(), pd.Series()

    df_last_cv = df.tail(cross_validation)
    df_describe = df_last_cv.describe()
    for column_ in df_describe.columns:
        serie_mean[column_] = df_describe[column_]["mean"]
        serie_std[column_] = df_describe[column_]["std"]
        serie_sem[column_] = df_last_cv.sem()[column_]

    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    df_sem = pd.DataFrame()

    df_mean = df_mean.append(serie_mean, ignore_index=True)
    df_std = df_std.append(serie_std, ignore_index=True)
    df_sem = df_sem.append(serie_sem, ignore_index=True)

    df_mean["cv_calculation"] = "mean"
    df_std["cv_calculation"] = "std"
    df_sem["cv_calculation"] = "sem"

    save_results_to_csv(df_result_csv_path, df_mean)
    save_results_to_csv(df_result_csv_path, df_std)
    save_results_to_csv(df_result_csv_path, df_sem)


def create_log_file(model_name: str) -> None:
    """
    log file config with datetime object containing current date and time

    :param model_name:
    """
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(f"results/run_{model_name}/{model_name}-{dt_string}.log"),
                                  logging.StreamHandler()])
    os.chmod(f"results/run_{model_name}/{model_name}-{dt_string}.log", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
