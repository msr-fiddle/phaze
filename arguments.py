# python imports
import argparse


def process_phaze_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phaze_micro_batch_size", nargs='+', type=int, required=False,
                        default=1, help="micro batch sizes to extract graph",)
    parser.add_argument("--phaze_model_names", type=str, nargs="*", required=True,
                        default=None, help="A list of model names from the supported group of models",)
    parser.add_argument("--force_reextract_model", type=bool, required=False,
                        default=False, help="Force the model extractor to reload the model",)
    parser.add_argument("--phaze_exec_type", type=str, required=False, default="run_solver",
                        help="phaze execution, run_solver runs the algorthmic solver, prepopulate estimates \
                        \\ creates a library of runtime estimates", choices=["run_solver", "prepopulate_estimates", "extract_graph"],)
    parser.add_argument("--phaze_sequence_length", type=int, required=False, default=64,
                        help="sequence length for language models. Bert uses 512 and GPT uses 2048 sequence \
                        \\ length. Default is 64 independent of the model",)
    parser.add_argument("--phaze_max_tmp_width", type=int, required=False, default=1,
                        help="Maximum tensor model parallel width. MegatronBert and MegatronGPT is 8 as per literature. \
                        \\ Other models do not support this option and should use the default. Default is 1.",)
    parser.add_argument("--phaze_hbm_size", nargs='+', type=int, required=False, default=32,
                        help="HBM sizes to explore in GB \
                        \\ Default is 32GB.",)

    args = parser.parse_args()

    return args
