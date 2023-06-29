import onnx
#from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto


def onnx_datatype_to_npType(data_type):
    if data_type == 1:      # true
        return np.float32
    elif data_type == 2:    # true
        return np.uint8
    elif data_type == 3:    # true
        return np.int8
    elif data_type == 4:
        return np.uint8
    elif data_type == 5:
        return np.uint8
    elif data_type == 6:   # true
        return np.int32
    elif data_type == 7:   # true
        return np.int64
    elif data_type == 8:
        return np.uint8
    elif data_type == 9:
        return np.bool8
    else:
        return np.float32


def get_raw_data(attr):
    data_type = attr.t.data_type
    data_dims = attr.t.dims
    data = np.frombuffer(attr.t.raw_data, dtype=onnx_datatype_to_npType(data_type))

    return data


def op_upgrade(model):
    graph = model.graph
    opset_version = model.opset_import[0].version

    if opset_version == 18:
        for i, node in enumerate(graph.node):
            # When opset>12， softmax attr: axis default is -1，but defult is 1 in tool
            if node.op_type == 'Softmax':
                axis = -1
                for attr in node.attribute:
                    if attr.name == 'axis':
                        if attr.t.raw_data:
                            axis = get_raw_data(attr)
                        else:
                            axis = attr.i

                new_softmax_node = helper.make_node(
                    "Softmax",
                    name=node.name,
                    inputs=node.input,
                    outputs=node.output,
                    axis=axis
                )

                graph.node.remove(node)
                graph.node.insert(i, new_softmax_node)
        return model

    for i, node in enumerate(graph.node):
        if node.op_type == 'Split' and opset_version < 13:  # upgrade split node to opset 18
            split = []
            axis = 0
            for attr in node.attribute:
                if attr.name == 'split':
                    if attr.t.raw_data:
                        split = get_raw_data(attr)
                    else:
                        split = attr.ints
                if attr.name == 'axis':
                    if attr.t.raw_data:
                        axis = get_raw_data(attr)
                    else:
                        axis = attr.i

            new_split = helper.make_tensor(node.name + '_split', TensorProto.INT64, [len(split)], split)
            node.input.append(node.name + '_split')

            split_node_opspet18 = helper.make_node(
                "Split",
                name=node.name,
                inputs=node.input,
                outputs=node.output,
                axis=axis,
            )

            graph.initializer.append(new_split)
            graph.node.remove(node)
            graph.node.insert(i, split_node_opspet18)

        if node.op_type == 'BatchNormalization' and opset_version < 14:  # upgrade BN node to opset 18
            if len(node.output) > 3:
                epsilon = 1e-5
                momentum = 0.9
                for attr in node.attribute:
                    if attr.name == 'epsilon':
                        if attr.t.raw_data:
                            epsilon = get_raw_data(attr)
                        else:
                            epsilon = attr.f
                    if attr.name == 'momentum':
                        if attr.t.raw_data:
                            momentum = get_raw_data(attr)
                        else:
                            momentum = attr.f

                bn_node_opspet18 = helper.make_node(
                    "BatchNormalization",
                    name=node.name,
                    inputs=node.input,
                    outputs=node.output[:3], # remove output : saved_mean and saved_var 
                    epsilon=epsilon,
                    momentum=momentum
                ) 
                graph.node.remove(node)
                graph.node.insert(i, bn_node_opspet18)

        if node.op_type == 'ReduceMin' and opset_version < 18:  # upgrade reduceMin node to opset 18， add axes to input
            axes = []
            keepdims = 1
            for attr in node.attribute:
                if attr.name == 'axes':
                    if attr.t.raw_data:
                        axes = get_raw_data(attr)
                    else:
                        axes = attr.ints
                if attr.name == 'keepdims':
                    if attr.t.raw_data:
                        keepdims = get_raw_data(attr)
                    else:
                        keepdims = attr.i

            new_axes = helper.make_tensor(node.name + '_axes', TensorProto.INT64, [len(axes)], axes)
            node.input.append(node.name + '_axes')

            reduceMin_node_opspet18 = helper.make_node(
                "ReduceMin",
                name=node.name,
                inputs=node.input,
                outputs=node.output,
                keepdims=keepdims,
            )

            graph.initializer.append(new_axes)
            graph.node.remove(node)
            graph.node.insert(i, reduceMin_node_opspet18)

        if node.op_type == 'ReduceMean' and opset_version < 18:  # upgrade reduceMean node to opset 18， add axes to input
            axes = []
            keepdims = 1
            for attr in node.attribute:
                if attr.name == 'axes':
                    if attr.t.raw_data:
                        axes = get_raw_data(attr)
                    else:
                        axes = attr.ints
                if attr.name == 'keepdims':
                    if attr.t.raw_data:
                        keepdims = get_raw_data(attr)
                    else:
                        keepdims = attr.i

            new_axes = helper.make_tensor(node.name + '_axes', TensorProto.INT64, [len(axes)], axes)
            node.input.append(node.name + '_axes')

            reduceMean_node_opspet18 = helper.make_node(
                "ReduceMean",
                name=node.name,
                inputs=node.input,
                outputs=node.output,
                keepdims=keepdims,
            )

            graph.initializer.append(new_axes)
            graph.node.remove(node)
            graph.node.insert(i, reduceMean_node_opspet18)

        if node.op_type == 'Squeeze' and opset_version < 13:  # upgrade Unsqueeze node to opset 13， add axes to input
            axes = [] 
            for attr in node.attribute:
                if attr.name == 'axes':
                    if attr.t.raw_data:
                        axes = get_raw_data(attr)
                    else:
                        axes = attr.ints

            new_axes = helper.make_tensor(node.name + '_axes', TensorProto.INT64, [len(axes)], axes)
            node.input.append(node.name + '_axes')

            Squeeze_node_opspet18 = helper.make_node(
                "Squeeze",
                name=node.name,
                inputs=node.input,
                outputs=node.output,
            )

            graph.initializer.append(new_axes)
            graph.node.remove(node)
            graph.node.insert(i, Squeeze_node_opspet18)

        if node.op_type == 'Unsqueeze' and opset_version < 13:  # upgrade Unsqueeze node to opset 13， add axes to input
            axes = [] 
            for attr in node.attribute:
                if attr.name == 'axes':
                    if attr.t.raw_data:
                        axes = get_raw_data(attr)
                    else:
                        axes = attr.ints

            new_axes = helper.make_tensor(node.name + '_axes', TensorProto.INT64, [len(axes)], axes)
            node.input.append(node.name + '_axes')

            Unsqueeze_node_opspet18 = helper.make_node(
                "Unsqueeze",
                name=node.name,
                inputs=node.input,
                outputs=node.output,
            )

            graph.initializer.append(new_axes)
            graph.node.remove(node)
            graph.node.insert(i, Unsqueeze_node_opspet18)
            
        # When opset>12， softmax attr: axis default is -1，but defult is 1 in tool
        if node.op_type == 'Softmax' and opset_version > 12:  
            axis = -1 
            for attr in node.attribute:
                if attr.name == 'axis':
                    if attr.t.raw_data:
                        axis = get_raw_data(attr)
                    else:
                        axis = attr.i

            new_softmax_node = helper.make_node(
                "Softmax",
                name=node.name,
                inputs=node.input,
                outputs=node.output,
                axis=axis
            )

            graph.node.remove(node)
            graph.node.insert(i, new_softmax_node)

    # new_model = helper.make_model(graph,
    #                              producer_name='onnx-example',
    #                              opset_imports=[onnx.helper.make_opsetid(domain="", version=18)])

    return model