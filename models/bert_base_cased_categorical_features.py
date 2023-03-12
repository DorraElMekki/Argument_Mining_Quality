import itertools
import random

from models.constants import *

from .utils import *


def train_test_bert_with_categorical_features(model_name="bert-base-cased") -> None:
    """
    train and evaluate bert model with categorical features.
    save the evaluation is df_results_csv file.
    :param model_name:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == "cuda", "GPU is required"

    create_folder_if_does_not_exist(f"{root}/results/run_{model_name}")
    create_log_file(model_name)

    logging.info(f"device: {device}")
    # for hyperparameter optimization, you can change all the loop lists.
    for weights_i, weights_j in itertools.product(range(8, 9), range(2, 3)):
        weights = [weights_i, 1, weights_j]
        for token in token_list:
            for textual_features_list in textual_features_lists:
                for numerical_features in numerical_features_list:
                    for lr in lr_list:
                        for num_epochs in num_epochs_list:
                            for batch_size in batch_size_list:
                                for fold_id in range(cross_validation):
                                    df_result = pd.DataFrame()
                                    num_extra_dims = len(numerical_features)
                                    dataset, num_labels = process_data_to_dataset_dict_type(
                                        dataset_csv_path,
                                        target_class,
                                        fold_id,
                                        token,
                                        textual_features_list,
                                        numerical_features,
                                    )
                                    tokenizer, train_dataloader, eval_dataloader = data_tokenization(
                                        dataset, model_name, batch_size
                                    )
                                    model, df_result = train_model(
                                        model_name,
                                        tokenizer,
                                        num_labels,
                                        token,
                                        lr,
                                        num_epochs,
                                        train_dataloader,
                                        device,
                                        weights,
                                        metrics_dict,
                                        df_result,
                                        num_extra_dims,
                                    )
                                    df_result = model_evaluation(
                                        device, model, weights, metrics_dict, df_result, eval_dataloader
                                    )
                                    df_result = save_model_description_in_df_results(
                                        df_result,
                                        model_name,
                                        target_class,
                                        weights,
                                        num_epochs,
                                        loss_function_name,
                                        task,
                                        fold_id,
                                        cross_validation,
                                        textual_features_list,
                                        numerical_features,
                                        lr,
                                        batch_size,
                                        token,
                                        dataset,
                                    )
                                    save_results_to_csv(df_result_csv_path, df_result)
                                add_mean_std_sem_for_the_previous_scores_resulting_from_cross_validation(
                                    cross_validation, df_result_csv_path
                                )


if __name__ == "__main__":
    train_test_bert_with_categorical_features()
