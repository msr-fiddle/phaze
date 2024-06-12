# internal phaze imports
from GraphExtractor import extract_graph
from Estimator import populate_estimates
from Solver import run_solver


def extract_only(model_names, max_tmp_width, micro_batch_size,
                 sequence_length, force_reextract_model,):
    # Every node has a corresponding estimates in a 3D matrix <TMP strategy,
    # core dimensions, and number of cores>
    for model_name in model_names:
        extract_graph(model_name, max_tmp_width, micro_batch_size,
                      sequence_length, force_reextract_model,)


def extract_and_populate(model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,):
    # Extract graph using Torch.fx
    print("Extracting graph for model: ", model_name, " ...")
    tmpc_models = extract_graph(
        model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,)

    print("Extracted graph for model: ", model_name, " ...")

    latency_estimates = populate_estimates(
        tmpc_models, max_tmp_width, micro_batch_size, sequence_length,)

    print("Populated estimates for model: ", model_name, " ...")

    return tmpc_models, latency_estimates


def extract_and_prepopulate(model_names, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,):
    # Every node has a corresponding estimates in a 3D matrix <TMP strategy,
    # core dimensions, and number of cores>
    for model_name in model_names:
        extract_and_populate(model_name, max_tmp_width,
                             micro_batch_size, sequence_length, force_reextract_model,)


def extract_and_solve(model_names, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model, activation_recomputation, hbm_size):
    models_info = {}

    for model_name in model_names:
        tmpc_models, latency_estimates = extract_and_populate(
            model_name, max_tmp_width, micro_batch_size, sequence_length, force_reextract_model,)

        models_info[model_name] = (tmpc_models, latency_estimates)

    print("Extracted graph and populated for all the models ...")

    return run_solver(models_info, micro_batch_size, max_tmp_width, sequence_length, activation_recomputation, hbm_size)
