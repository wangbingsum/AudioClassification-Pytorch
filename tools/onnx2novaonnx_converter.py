import onnx
#from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto
from onnx import version_converter
from onnx import numpy_helper

import sys
import argparse
from load_save_model_shm import *
from opset_upgrade import op_upgrade


ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

SUPPORTED_OP_TYPE_LIST = [
'Abs',
'Add',
'AveragePool',
'BatchNormalization',
'Clip',
'Conv',
'ConvTranspose',
'Concat',
'Flatten',
'Gemm',
'GlobalAveragePool',
'GlobalMaxPool',
'LeakyRelu',
'LSTM',
'MatMul',
'Max',
'MaxPool',
'MaxRoiPool',
'Mul',
'Pad',
'PRelu',
'ReduceMean',
'Relu',
'Resize',
'Sigmoid',
'Softmax',
'Sub',
'Tanh',
'Transpose',
'Upsample',
'Reshape',
'Slice',
'Split',
'Neg',
'Sub',
'Tanh',
'Sqrt',
'Exp',
'Div',
'Log',
'Pow',
'Sin',
'Floor',
'Round',
'Squeeze',
'Unsqueeze',
'LayerNormalization'
]


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='input model path')
    parser.add_argument('--output', '-o', type=str, required=True, help='output model path')
    parser.add_argument('--skip_fuse_bn', '-s', type=int, default = 1 ,help='skip bn folding, default = 1(True)')
    parser.add_argument('--skip_onnx_sim', '-m', type=int, default = 0 ,help='skip onnx simplifier, default = 0(False)')
    parser.add_argument('--skip_modify_idx', '-d', type=int, default = 0 ,help='skip modify layer_idx, default = 0(False)')
    parser.add_argument("-s1", "--input_model_size", type = int, help="Set input model size")
    parser.add_argument("-f", "--no_file_op", type = int, choices=[0, 1, 2], default = 0, help="Set no_file_op, 1:read model from shm1, 2:read model from shm2")
    parser.add_argument("-n", "--name", help="Set file name")
    parser.add_argument("-v", "--verbose", type = int, choices=[0, 1, 2, 3, 4], default = 2, help="Set verbose level to <number>, default is 2. (0: fatal, 1: error, 2: warning, 3: index, 4: user)")

    return parser.parse_args()


def onnx_attribute_to_dict(onnx_attr):
    #print(onnx_attr)
    if onnx_attr.HasField('name'):
        name = getattr(onnx_attr, 'name')
        #print(name)

    if onnx_attr.HasField('t'):
        return name, numpy_helper.to_array(getattr(onnx_attr, 't'))

    for attr_type in ['f', 'i', 's']:
        if onnx_attr.HasField(attr_type):
            return name, getattr(onnx_attr, attr_type)

    for attr_type in ['floats', 'ints', 'strings']:
        if getattr(onnx_attr, attr_type):
            return name, list(getattr(onnx_attr, attr_type))

def add_input_from_initializer(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)

def ReplaceUpsampleWithResize(onnx_model):

    graph = onnx_model.graph

    for i in range(len(graph.node)):
        if graph.node[i].op_type == 'Upsample':
            old_node = graph.node[i]
            roi = numpy_helper.from_array(np.empty([0], dtype=np.float32), old_node.name + "_roi")
            onnx_model.graph.initializer.append(roi)
            roi_value_info = helper.make_tensor_value_info(old_node.name + "_roi", onnx.TensorProto.FLOAT, [0])
            onnx_model.graph.value_info.append(roi_value_info)
            inputs = [old_node.input[0], old_node.name + "_roi", old_node.input[1]]
            mode_string = ''
            for attr in graph.node[i].attribute:
                if attr.name == 'mode':
                    mode_string = attr.s
            new_node = onnx.helper.make_node(
                "Resize",
                coordinate_transformation_mode="asymmetric",
                cubic_coeff_a=-0.75,
                mode=mode_string,
                nearest_mode="floor",
                inputs=inputs,
                outputs=old_node.output
            )
            graph.node.remove(old_node)
            graph.node.insert(i, new_node)
            

def ReplaceThresholdWithPthThreshold(onnx_model):
    graph = onnx_model.graph
    for i in range(len(graph.node)):
        if graph.node[i].op_type == 'threshold':
            old_node = graph.node[i]
            threshold_name = old_node.input[1]
            replace_val_name = old_node.input[2]
            for init in graph.initializer:
                if init.name == threshold_name:
                    threshold_data = init.float_data[0]
                elif init.name == replace_val_name:
                    replace_val_data = init.float_data[0]

            new_node = onnx.helper.make_node(
                "PthThreshold",
                inputs = [old_node.input[0]],
                outputs = old_node.output,
                threshold = threshold_data,
                replace_val = replace_val_data
            )
            graph.node.remove(old_node)
            graph.node.insert(i, new_node)


def check_shapes(onnx_model, verbose):
    names = []
    for input_tensor in onnx_model.graph.input:
        names.append(input_tensor.name)
    for output_tensor in onnx_model.graph.output:
        names.append(output_tensor.name)
    for init_tensor in onnx_model.graph.initializer:
        names.append(init_tensor.name)
    for value in onnx_model.graph.value_info:
        names.append(value.name)

    for node in onnx_model.graph.node:
        outputs = node.output
        for output in outputs:
            if output not in names:
                fake_value_info = helper.make_tensor_value_info(output, TensorProto.FLOAT, [-1,-1,-1,-1])
                onnx_model.graph.value_info.append(fake_value_info)
                if verbose >= 2:
                    print("Warning: Shape checking error. Node: %s Type: %s, cannot get output shape, please check the attribute or onnx version." % (node.name, node.op_type))


def onnx_datatype_to_npType(data_type):
    if data_type == 1:     
        return np.float32
    elif data_type == 2:   
        return np.uint8
    elif data_type == 3:   
        return np.int8
    elif data_type == 4:
        return np.uint16
    elif data_type == 5:
        return np.int16
    elif data_type == 6:  
        return np.int32
    elif data_type == 7: 
        return np.int64
    elif data_type == 8:  # string
        return str
    elif data_type == 9:
        return bool
    elif data_type == 10:
        return np.float16
    elif data_type == 11:
        return np.float64
    elif data_type == 12:
        return np.uint32
    elif data_type == 13:
        return np.uint64
    else:
        return np.float32


def get_onnx_tensortype(data_type):
    if data_type == 1 or data_type == 0:
        tensor_type = onnx.TensorProto.FLOAT
    elif data_type == 2:
        tensor_type = onnx.TensorProto.UINT8
    elif data_type == 3:
        tensor_type = onnx.TensorProto.INT8
    elif data_type == 4:
        tensor_type = onnx.TensorProto.UINT16
    elif data_type == 5:
        tensor_type = onnx.TensorProto.INT16
    elif data_type == 6:
        tensor_type = onnx.TensorProto.INT32
    elif data_type == 7:
        tensor_type = onnx.TensorProto.INT64
    elif data_type == 8:
        tensor_type = onnx.TensorProto.STRING
    elif data_type == 9:
        tensor_type = onnx.TensorProto.BOOL
    elif data_type == 10:
        tensor_type = onnx.TensorProto.FLOAT16
    elif data_type == 11:
        tensor_type = onnx.TensorProto.DOUBLE
    elif data_type == 12:
        tensor_type = onnx.TensorProto.UINT32
    elif data_type == 13:
        tensor_type = onnx.TensorProto.UINT64
    else:
        tensor_type = 0

    return tensor_type


def Constant_to_initializer(onnxmodel):
    graph = onnxmodel.graph
    delete = []
    for i in range(len(graph.node)):
        if graph.node[i].op_type=="Constant":
            data_type = graph.node[i].attribute[0].t.data_type
            data_dims = graph.node[i].attribute[0].t.dims

            if graph.node[i].attribute[0].t.raw_data:
                data = np.frombuffer(graph.node[i].attribute[0].t.raw_data, dtype=onnx_datatype_to_npType(data_type))
            elif data_type == 0:
                data = np.array(graph.node[i].attribute[0].f)
            
            tensor_type = get_onnx_tensortype(data_type)
            assert tensor_type != 0, "Unupported constant data type. Node: %s." % (graph.node[i].name)

            data = data.flatten()
            p_t = helper.make_tensor(graph.node[i].output[0], tensor_type, dims=data_dims, vals=data, raw=False)
            delete.append(graph.node[i])
            graph.initializer.append(p_t)
    for oldnode in delete:
        graph.node.remove(oldnode)

def modify_layer_dix(graph):
    outputs = graph.output
    outputs_dict = {}
    for i, output in enumerate(outputs):
        for j, node in enumerate(graph.node):
            if output.name in node.output:
                # output_idx : node_idx, layer_idx
                outputs_dict[i] = [j, j]

    for i in range(len(outputs_dict)):
        min_index = i  
        # find min_index
        for j in range(i+1, len(outputs_dict)):
            if outputs_dict[j][1] < outputs_dict[min_index][1]:
                min_index = j

        if min_index != i:
                # exchange layer idx
                for k, attr in enumerate(graph.node[outputs_dict[i][0]].attribute):
                    if attr.name == 'layer_idx':
                        new_layer_idx = onnx.helper.make_attribute("layer_idx", outputs_dict[min_index][1])
                        del graph.node[outputs_dict[i][0]].attribute[k]
                        graph.node[outputs_dict[i][0]].attribute.extend([new_layer_idx])
                        break

                for k, attr in enumerate(graph.node[outputs_dict[min_index][0]].attribute):
                    if attr.name == 'layer_idx':
                        new_layer_idx = onnx.helper.make_attribute("layer_idx", outputs_dict[i][1])
                        del graph.node[outputs_dict[min_index][0]].attribute[k]
                        graph.node[outputs_dict[min_index][0]].attribute.extend([new_layer_idx])
                        break

                # if graph.node[1].attribute
                outputs_dict[i][1], outputs_dict[min_index][1] = outputs_dict[min_index][1], outputs_dict[i][1]

    return graph


def save_onnx_model(onnx_model, no_file_op, out_model_path, name, verbose):
    if no_file_op == 1:
        onnx_model = saveonnxmodel_shm(onnx_model, name, verbose)
    elif no_file_op == 2:
        onnx_model = saveonnxmodel_shm2(onnx_model, name, verbose)
    else:
        onnx.save(onnx_model, out_model_path)


def to_nova_onnx(onnx_model, skip_fuse_bn, skip_onnx_sim, skip_modify_idx, verbose):

    if onnx_model.producer_name == 'Novatek NovaOnnx Converter' or onnx_model.producer_name == 'Novatek Caffe2Onnx Converter':
        print("This model is already a nova onnx model, skip the conversion process...")
        return
    
    # check input shape
    for input in onnx_model.graph.input:
        input_shape = input.type.tensor_type.shape.dim
        for d in input_shape:
            if d.dim_value <= 0 and d.dim_param == '':
                assert (False), "Each dimension of input shape must greater than zero, illegal input name = %s"% input.name
    Constant_to_initializer(onnx_model)
    # convert model
    add_input_from_initializer(onnx_model)
    
    has_caffe_op = 0
    has_custom_op = 0
    has_pytorch_op = 0
    for node in onnx_model.graph.node:
        if node.domain == 'caffe_ops':
            has_caffe_op = 1
            if node.op_type == 'Crop':
                node.op_type = 'CaffeCrop'
        elif node.domain == 'org.pytorch.aten' :
            has_pytorch_op = 1
            if node.op_type == 'threshold':
                ReplaceThresholdWithPthThreshold(onnx_model)
            else:
                assert False, "Pytorch op is node support now!, op name = %s, type = %s"% (node.name, node.op_type)
        elif node.domain != '' and node.domain != 'ai.onnx':
            has_custom_op = 1
    if has_custom_op or has_caffe_op or has_pytorch_op:

        #get all value_info and output name
        tensor_names = []
        for vi in onnx_model.graph.value_info:
            tensor_names.append(vi.name)
        for output in onnx_model.graph.output:
            tensor_names.append(output.name)
        
        # Add missing tensor_value_info (fake shape)
        for i in range(len(onnx_model.graph.node)):
            for output in onnx_model.graph.node[i].output:
                if output not in tensor_names:
                    if onnx_model.graph.node[i].op_type == "Gemm" or onnx_model.graph.node[i].op_type == "Flatten":
                        fake_value_info = helper.make_tensor_value_info(output, TensorProto.FLOAT, [-1,-1])
                    else:
                        fake_value_info = helper.make_tensor_value_info(output, TensorProto.FLOAT, [-1,-1,-1,-1])
                    tensor_names.append(output)
                    onnx_model.graph.value_info.append(fake_value_info)
    
    else:
        # convert model to opset 12
        if onnx_model.opset_import[0].version != 12:
            if onnx_model.opset_import[0].version < 8:
                assert (False), ": Opset version of the input model is %d, onnx2novaonnx tool only support Opset version 8 ~ 18."% onnx_model.opset_import[0].version
            if verbose >= 2:
                # print("Warning: Opset version of the input model is {}, Novaic tool support Opset version 12.".format(onnx_model.opset_import[0].version))
                print("Convert from Opset version {} to Opset version 12.".format(onnx_model.opset_import[0].version))
                
            if onnx_model.opset_import[0].version < 12:
                for i in range(len(onnx_model.graph.value_info)):  # onnx = 1.9.0, need to clean value info before convert version
                    del onnx_model.graph.value_info[0]
                onnx_model = version_converter.convert_version(onnx_model, 12)
            
            #version_converter can not convert upsample(deprecated in opset 12), convert it to resize 
            ReplaceUpsampleWithResize(onnx_model)
        
        if skip_onnx_sim:
            onnx_model = shape_inference.infer_shapes(onnx_model)
            check_shapes(onnx_model, verbose)
        else:
            # apply onnx simplify
            from onnxsim import simplify
            onnx_model, check = simplify(onnx_model, skip_fuse_bn = skip_fuse_bn)

            assert check, "Simplified ONNX model could not be validated"
            
        # convert to opset 18
        onnx_model = op_upgrade(onnx_model)

        for i in range(len(onnx_model.graph.node)):
            if onnx_model.graph.node[i].op_type not in SUPPORTED_OP_TYPE_LIST:
                print("Warning: novaic tool can't support ", onnx_model.graph.node[i].op_type)

    graph = onnx_model.graph

        
    init_name_list = []
    for initializer in graph.initializer:
        init_name_list.append(initializer.name)

    input_name_list = []
    for input in graph.input:
        if input.name not in init_name_list:
            input_name_list.append(input.name)
            input.name = input.name.replace('/','_').replace(':','_').replace('@','_').replace(' ','_').replace('|','_').replace('*','_').replace('?','_')

    for node in graph.node:
        for i in range(len(node.input)):
            if node.input[i] in input_name_list:
                node.input[i] = node.input[i].replace('/','_').replace(':','_').replace('@','_').replace(' ','_').replace('|','_').replace('*','_').replace('?','_')

    name_dict = {}
            
    #modify Conv weight name
    for i in range(len(graph.node)):
        if graph.node[i].op_type == 'Conv':
            if graph.node[i].input[1] in init_name_list:
                name_dict.setdefault(graph.node[i].input[1], graph.node[i].op_type + "_" + graph.node[i].input[1] + "_W")
                graph.node[i].input[1] = graph.node[i].op_type + "_" + graph.node[i].input[1] + "_W"
            if len(graph.node[i].input) > 2:
                if graph.node[i].input[2] in init_name_list:
                    name_dict.setdefault(graph.node[i].input[2],  graph.node[i].op_type + "_" + graph.node[i].input[2] + "_B")
                    graph.node[i].input[2] = graph.node[i].op_type + "_" + graph.node[i].input[2] + "_B"

      
        #modify output tensor_name to (node_name)_Y
        for k in range(len(graph.node[i].input)):
            if graph.node[i].input[k] in name_dict:
                graph.node[i].input[k] = name_dict[graph.node[i].input[k]]
        for l in range(len(graph.node[i].output)):
            name_dict.setdefault(graph.node[i].output[l], graph.node[i].op_type + "_" + graph.node[i].output[l] + "_Y")
            graph.node[i].output[l] = graph.node[i].op_type + "_" + graph.node[i].output[l] + "_Y"

        # Add layer_id attribute for each node
        new_attr = helper.make_attribute("layer_idx", i)
        graph.node[i].attribute.append(new_attr)
        
        #modify Conv weight name
        if graph.node[i].op_type == 'AveragePool' or graph.node[i].op_type == 'MaxPool':
            new_attr = helper.make_attribute("pool_at_pad", 1)
            graph.node[i].attribute.append(new_attr)

    #print(graph.value_info)
    #modify graph output tensor_name to (node_name)_Y
    for m in range(len(graph.output)):
        if graph.output[m].name in name_dict:
            graph.output[m].name = name_dict[graph.output[m].name]
            
    #modify value info name
    for n in range(len(graph.value_info)):
        if graph.value_info[n].name in name_dict:
            graph.value_info[n].name = name_dict[graph.value_info[n].name]

    #modify input name
    for o in range(len(graph.input)):
        if graph.input[o].name in name_dict:
            graph.input[o].name = name_dict[graph.input[o].name] 
            
    #modify initializer name
    for p in range(len(graph.initializer)):
        if graph.initializer[p].name in name_dict:
            graph.initializer[p].name = name_dict[graph.initializer[p].name] 
    
    if not skip_modify_idx:
        graph = modify_layer_dix(graph)

    onnx_model.producer_name = 'Novatek NovaOnnx Converter'
    onnx_model.producer_version = '1.0'

    if verbose >= 2:
        print("Convertered NOVA ONNX model done")

    return onnx_model


if __name__ == '__main__':
    args = process_command()
    if args.no_file_op == 1:
        onnx_model = loadonnxmodel_shm(args.input_model_size, args.name, args.verbose)
    elif args.no_file_op == 2:
        onnx_model = loadonnxmodel_shm2(args.input_model_size, args.name, args.verbose)
    else:
        onnx_model = onnx.load(args.input)
        print("input:", args.input)
        print("output:", args.output)

    onnx_model = to_nova_onnx(onnx_model, args.skip_fuse_bn, args.skip_onnx_sim, args.skip_modify_idx, args.verbose)
    save_onnx_model(onnx_model, args.no_file_op, args.output, args.name, args.verbose)
