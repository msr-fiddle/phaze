# internal phaze imports
from GraphExtractor.models import BertIR
from GraphExtractor.models import GptIR
from GraphExtractor.models import OptIR
from GraphExtractor.models import LlamaIR
from GraphExtractor.models import BaseModelIR
from .utils import load_obj_from_file, check_model
from .utils import (bert_models, gpt_models, opt_models, llama_models, language_models,
                    )

# python imports
from math import log2

supported_models = language_models


def get_tmp_widths(max_tmp_width=1):
    tmp_widths = []

    if max_tmp_width == 1:
        return [1]
    max_log_width_iter = int(log2(max_tmp_width)) + 1
    for i in range(0, max_log_width_iter):
        tmp_widths.append(2**i)
    return tmp_widths


def extract_graph(model_name, max_tmp_width=1, micro_batch_size=1, sequence_length=64, force_reextract_model=False,):
    def extract_model_from_file(tmp_width):
        if not force_reextract_model:
            model = load_obj_from_file(
                model_name, micro_batch_size, tmp_width, sequence_length,)

            if check_model(model, BaseModelIR, supported_models):
                model_memory = model.phazegraph.get_memory_footprint()
                print(model_memory.parameter_size)
                return model

    model_name = model_name.lower()

    if model_name in bert_models:
        tmp_widths = get_tmp_widths(max_tmp_width)
        bertmodels = []

        for width in tmp_widths:
            bert = extract_model_from_file(width)
            if bert is None:
                bert = BertIR(model_name, width)
                bert.extract_model_graph(
                    micro_batch_size, sequence_length, force_reextract_model)
            bertmodels.append(bert)
        return bertmodels

    elif model_name in gpt_models:
        tmp_widths = get_tmp_widths(max_tmp_width)
        gptmodels = []

        for width in tmp_widths:
            if (model_name == "megatrongpt3" and width < 4):
                continue
            gpt = extract_model_from_file(width)
            if gpt is None:
                gpt = GptIR(model_name, width)
                gpt.extract_model_graph(
                    micro_batch_size, sequence_length, force_reextract_model)
            gptmodels.append(gpt)
        return gptmodels

    elif model_name in opt_models:
        opt = extract_model_from_file(max_tmp_width)

        if opt:
            return [opt]

        opt = OptIR(model_name)
        opt.extract_model_graph(
            micro_batch_size, sequence_length, force_reextract_model)

        return [opt]

    elif model_name in llama_models:
        llama = extract_model_from_file(max_tmp_width)

        if llama:
            return [llama]

        llama = LlamaIR(model_name)
        llama.extract_model_graph(
            micro_batch_size, sequence_length, force_reextract_model)

        return [llama]

    else:
        raise ValueError(
            "Only following models '{}' are currently imported.".format(supported_models))
