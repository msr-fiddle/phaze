
op_to_compute_dict = {
    "MkldnnConvolution": "Tensor Core",
    "CudnnConvolution": "Tensor Core",
    "ThnnConv2D": "Tensor Core",
    "CudnnRnn": "Tensor Core",
    "Addmm": "Tensor Core",
    "Linear": "Tensor Core",
    "Mm": "Tensor Core",
    "Bmm": "Tensor Core",
    "fused ~~ MkldnnConvolution": "Tensor Core + Vector Core",
    "fused ~~ CudnnConvolution": "Tensor Core + Vector Core",
    "fused ~~ ThnnConv2D": "Tensor Core + Vector Core",
    "fused ~~ CudnnRnn": "Tensor Core + Vector Core",
    "fused ~~ Addmm": "Tensor Core + Vector Core",
    "fused ~~ Linear": "Tensor Core + Vector Core",
    "fused ~~ Mm": "Tensor Core + Vector Core",
    "fused ~~ Bmm": "Tensor Core + Vector Core",
    "CudnnBatchNorm": "Vector Core",
    "NativeBatchNorm": "Vector Core",
    "NativeLayerNorm": "Vector Core",
    "CudnnLayerNorm": "Vector Core",
    "Relu": "Vector Core",
    "Tanh": "Vector Core",
    "Cos": "Vector Core",
    "Sin": "Vector Core",
    "Hardtanh": "Vector Core",
    "Hardswish": "Vector Core",
    "Hardsigmoid": "Vector Core",
    "Sigmoid": "Vector Core",
    "Silu": "Vector Core",
    "Add": "Vector Core",
    "Mul": "Vector Core",
    "Div": "Vector Core",
    "Sub": "Vector Core",
    "Pow": "Vector Core",
    "Sqrt": "Vector Core",
    "Addcmul": "Vector Core",
    "Neg": "Vector Core",
    "fused ~~ Add": "Vector Core",
    "fused ~~ Mul": "Vector Core",
    "fused ~~ Div": "Vector Core",
    "fused ~~ Sub": "Vector Core",
    "fused ~~ Pow": "Vector Core",
    "fused ~~ Sqrt": "Vector Core",
    "Mean": "Vector Core",
    "Sum": "Nop",
    "MaxPool2DWithIndices": "Vector Core",
    "AdaptiveAveragePool2D": "Vector Core",
    "AdaptiveAvgPool2D": "Vector Core",
    "AvgPool2D": "Vector Core",
    "NativeDropout": "Vector Core",
    "FusedDropout": "Vector Core",
    "LogSoftmax": "Vector Core",
    "Softmax": "Vector Core",
    "NllLoss": "Vector Core",
    "Erf": "Vector Core",
    "MaskedFill": "Vector Core",
    "Embedding": "Nop",
    "View": "Nop",
    "T": "Nop",
    "Transpose": "Nop",
    "UnsafeView": "Nop",
    "Permute": "Nop",
    "AccumulateGrad": "Nop",
    "source": "Nop",
    "sink": "Nop",
    "SOURCE": "Nop",
    "SINK": "Nop",
    "SRC": "Nop",
    "SNK": "Nop",
    "dummy": "Nop",
    "Gather": "Nop",
    "Revert_varlen": "Nop",
    "Cat": "Nop",
    "Clone": "Nop",
    "Unsqueeze": "Nop",
    "Expand": "Nop",
    "Norm": "Nop",
    "Alias": "Nop",
    "Slice": "Nop",
    "Select": "Nop",
    "Split": "Nop",
    "SWhere": "Nop",
    "Copy": "Nop",
    "Scatter": "Nop",
    "Index": "Nop",
    "Squeeze": "Nop",
    "CopySlices": "Nop",
    "PackPaddedSequence": "Nop",
    "Stack": "Nop",
    "UnsafeSplit": "Nop",
    "Unbind": "Nop",
    "AllReduceFwd": "Vector Core",
    "AllReduceBwd": "Vector Core",
    "AllReduce": "Vector Core",
}

op_to_opr_dict = {
    "MkldnnConvolution": "complex_out_fusable_opr",
    "CudnnConvolution": "complex_out_fusable_opr",
    "ThnnConv2D": "complex_out_fusable_opr",
    "CudnnRnn": "complex_out_fusable_opr",
    "Addmm": "complex_out_fusable_opr",
    "Linear": "complex_out_fusable_opr",
    "Mm": "complex_out_fusable_opr",
    "Bmm": "complex_out_fusable_opr",
    "fused ~~ MkldnnConvolution": "complex_out_fusable_opr",
    "fused ~~ CudnnConvolution": "complex_out_fusable_opr",
    "fused ~~ ThnnConv2D": "complex_out_fusable_opr",
    "fused ~~ CudnnRnn": "complex_out_fusable_opr",
    "fused ~~ Addmm": "complex_out_fusable_opr",
    "fused ~~ Linear": "complex_out_fusable_opr",
    "fused ~~ Mm": "complex_out_fusable_opr",
    "fused ~~ Bmm": "complex_out_fusable_opr",
    "CudnnBatchNorm": "norm_opr",
    "NativeBatchNorm": "norm_opr",
    "NativeLayerNorm": "norm_opr",
    "CudnnLayerNorm": "norm_opr",
    "Relu": "activation_opr",
    "Tanh": "activation_opr",
    "Hardtanh": "activation_opr",
    "Cos": "activation_opr",
    "Sin": "activation_opr",
    "Hardswish": "activation_opr",
    "Hardsigmoid": "activation_opr",
    "Sigmoid": "activation_opr",
    "Silu": "activation_opr",
    "Add": "injective_opr",
    "Mul": "injective_opr",
    "Div": "injective_opr",
    "Sub": "injective_opr",
    "Pow": "injective_opr",
    "Sqrt": "injective_opr",
    "Addcmul": "injective_opr",
    "Neg": "injective_opr",
    "fused ~~ Add": "injective_opr",
    "fused ~~ Mul": "injective_opr",
    "fused ~~ Div": "injective_opr",
    "fused ~~ Sub": "injective_opr",
    "fused ~~ Pow": "injective_opr",
    "fused ~~ Sqrt": "injective_opr",
    "fused ~~ Addcmul": "injective_opr",
    "Mean": "reduction_opr",
    "Sum": "reduction_opr",
    "MaxPool2DWithIndices": "pooling_opr",
    "AdaptiveAveragePool2D": "pooling_opr",
    "AdaptiveAvgPool2D": "pooling_opr",
    "AvgPool2D": "pooling_opr",
    "NativeDropout": "dropout_opr",
    "FusedDropout": "dropout_opr",
    "LogSoftmax": "softmax_opr",
    "Softmax": "softmax_opr",
    "NllLoss": "loss_opr",
    "Erf": "loss_opr",
    "Embedding": "memory_opr",
    "View": "transformation_opr",
    "T": "transformation_opr",
    "Transpose": "transformation_opr",
    "UnsafeView": "transformation_opr",
    "Permute": "transformation_opr",
    "MaskedFill": "transformation_opr",
    "AccumulateGrad": "Nop",
    "source": "Nop",
    "sink": "Nop",
    "SOURCE": "Nop",
    "SINK": "Nop",
    "SRC": "Nop",
    "SNK": "Nop",
    "dummy": "Nop",
    "Gather": "Nop",
    "Revert_varlen": "Nop",
    "Cat": "Nop",
    "Clone": "Nop",
    "Unsqueeze": "Nop",
    "Expand": "Nop",
    "Norm": "Nop",
    "Alias": "Nop",
    "Slice": "Nop",
    "Select": "Nop",
    "Split": "Nop",
    "SWhere": "Nop",
    "Copy": "Nop",
    "Scatter": "Nop",
    "Index": "Nop",
    "Squeeze": "Nop",
    "CopySlices": "Nop",
    "PackPaddedSequence": "Nop",
    "Stack": "Nop",
    "UnsafeSplit": "Nop",
    "Unbind": "Nop",
    "AllReduceFwd": "collective_opr",
    "AllReduceBwd": "collective_opr",
    "AllReduce": "collective_opr",
}

complex_out_fusable_opr = {
    "MkldnnConvolution": "Tensor Core",
    "CudnnConvolution": "Tensor Core",
    "ThnnConv2D": "Tensor Core",
    "CudnnRnn": "Tensor Core",
    "Addmm": "Tensor Core",
    "Linear": "Tensor Core",
    "CudnnRnn": "Tensor Core",
    "Mm": "Tensor Core",
    "Bmm": "Tensor Core",
}

activation_opr = {
    "Relu": "Vector Core",
    "Tanh": "Vector Core",
    "Cos": "Vector Core",
    "Sin": "Vector Core",
    "Hardtanh": "Vector Core",
    "Hardswish": "Vector Core",
    "Hardsigmoid": "Vector Core",
    "Sigmoid": "Vector Core",
    "Silu": "Vector Core",
}

norm_opr = {
    "CudnnBatchNorm": "Vector Core",
    "NativeBatchNorm": "Vector Core",
    "NativeLayerNorm": "Vector Core",
    "CudnnLayerNorm": "Vector Core",
}

injective_opr = {
    "Add": "Vector Core",
    "Mul": "Vector Core",
    "Div": "Vector Core",
    "Sub": "Vector Core",
    "Pow": "Vector Core",
    "Sqrt": "Vector Core",
    "Addcmul": "Vector Core",
    "Neg": "Vector Core",
}

reduction_opr = {"Mean": "Vector Core", "Sum": "Nop"}

pooling_opr = {
    "MaxPool2DWithIndices": "Vector Core",
    "AdaptiveAveragePool2D": "Vector Core",
    "AdaptiveAvgPool2D": "Vector Core",
    "AvgPool2D": "Vector Core",
}

dropout_opr = {"NativeDropout": "dropout_opr", "FusedDropout": "dropout_opr"}

softmax_opr = {"LogSoftmax": "Vector Core", "Softmax": "Vector Core"}

loss_opr = {
    "NllLoss": "Vector Core",
    "Erf": "Vector Core",
}

memory_opr = {"Embedding": "memory_opr"}

transformation_opr = {
    "View": "transformation_opr",
    "T": "transformation_opr",
    "Transpose": "transformation_opr",
    "UnsafeView": "transformation_opr",
    "Permute": "transformation_opr",
    "MaskedFill": "transformation_opr",
}

collective_opr = {
    "AllReduceFwd": "collective_opr",
    "AllReduceBwd": "collective_opr",
    "AllReduce": "collective_opr",
}

fwd_pe_cycles_dict = {
    "CudnnBatchNorm": 1,
    "NativeBatchNorm": 1,
    "NativeLayerNorm": 1,
    "CudnnLayerNorm": 1,
    "Relu": 1,
    "Tanh": 27,
    "Hardtanh": 27,
    "Cos": 13,
    "Sin": 13,
    "Hardswish": 14,
    "Swiss": 14,
    "Hardsigmoid": 12,
    "Sigmoid": 12,
    "Silu": 12,
    "Add": 1,
    "Mul": 1,
    "Div": 10,
    "Sub": 1,
    "Pow": 15,
    "Sqrt": 15,
    "Addcmul": 1,
    "Neg": 1,
    "Mean": 10,
    "Sum": 0,
    "NativeDropout": 2,
    "FusedDropout": 2,
    "LogSoftmax": 13,
    "Softmax": 10,
    "NllLoss": 2,
    "Erf": 15,
    "MaskedFill": 1,
}

bwd_pe_cycles_dict = {
    "CudnnBatchNorm": 1,
    "NativeBatchNorm": 1,
    "NativeLayerNorm": 1,
    "CudnnLayerNorm": 1,
    "Relu": 1,
    "Tanh": 2,
    "Cos": 2,
    "Sin": 2,
    "Hardtanh": 2,
    "Hardswish": 15,
    "Swiss": 15,
    "Hardsigmoid": 2,
    "Sigmoid": 2,
    "Silu": 2,
    "Add": 1,
    "Mul": 1,
    "Div": 10,
    "Sub": 1,
    "Pow": 16,
    "Sqrt": 16,
    "Addcmul": 1,
    "Neg": 1,
    "Mean": 10,
    "Sum": 0,
    "NativeDropout": 1,
    "FusedDropout": 1,
    "LogSoftmax": 2,
    "Softmax": 2,
    "NllLoss": 2,
    "Erf": 15,
    "MaskedFill": 1,
}


def get_compute_unit(op):
    return op_to_compute_dict[op]


def get_opr_type(op):
    return op_to_opr_dict[op]


def get_fwd_pe_cycles(op):
    return fwd_pe_cycles_dict[op]


def get_bwd_pe_cycles(op):
    return bwd_pe_cycles_dict[op]