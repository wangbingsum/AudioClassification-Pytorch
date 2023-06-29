import onnx
from onnx import helper, numpy_helper
import numpy as np

class NodeRef():
    def __init__(self, onnx_node, prev_nodes=[], next_nodes=[]):
        self.onnx_node = onnx_node
        self.prev_nodes = prev_nodes
        self.next_nodes = next_nodes
        self.input_shapes = None
        self.output_shapes = None
        self.input_weights = None

    def __getattr__(self, attr):
        return getattr(self.onnx_node, attr)

    def setPrevNodes(self, prev_nodes):
        self.prev_nodes = prev_nodes

    def setNextNodes(self, next_nodes):
        self.next_nodes = next_nodes

    def setInputShapes(self, input_shapes):
        self.input_shapes = input_shapes

    def setOutputShapes(self, output_shapes):
        self.output_shapes = output_shapes

    def setInputWeights(self, input_weights):
        self.input_weights = input_weights

    def getPrevNodes(self):
        return self.prev_nodes

    def getNextNodes(self):
        return self.next_nodes

    def getInputShapes(self):
        return self.input_shapes

    def getOutputShapes(self):
        return self.output_shapes

    def getInputWeights(self):
        return self.input_weights


class GraphReplace():
    def __init__(self):
        self.graph_node_types = None
        self.graph_nodes = None

    def __call__(self, node, out_layer_list, replaced_list, kwargs=None):
        if kwargs is None:
            kwargs = {}

        node_list = []
        is_success = self.findStructureByType(node, self.graph_nodes[0], node_list)
        if is_success:
            is_success = self.chkLayersByAttr(node_list, kwargs)
            if is_success:
                self.modifyValueInfo(node_list, kwargs)
                out_list = self.replaceLayers(node_list, kwargs)
                out_layer_list.extend(out_list)
                replaced_list.extend(node_list)
                return True
        
        return False

    def initNodeGraph(self):
        raise NotImplementedError

    def chkLayersByAttr(self, node_list, kwargs):
        raise NotImplementedError

    def modifyValueInfo(self, node_list, kwargs):
        raise NotImplementedError

    def replaceLayers(self, node_list, kwargs):
        raise NotImplementedError

    def replacePrevLayerOut(self, prev_layer, prev_trg_names, fused_layer):
        tmp_list = []
        tmp_loc = []
        add_flag = True
        if prev_layer:
            for idx, i in enumerate(prev_layer.lparams['out']):
                if i[0] and i[0].name in prev_trg_names:
                    if add_flag:
                        tmp_list.append([(fused_layer), i[1]])
                        tmp_loc.append(prev_layer.getTileInfo().out_loc[idx])
                        add_flag = False
                else:
                    tmp_list.append([(i[0]), i[1]])
                    tmp_loc.append(prev_layer.getTileInfo().out_loc[idx])

            prev_layer.lparams['out'] = tmp_list
            prev_layer.getTileInfo().out_loc = tmp_loc

    def replaceNextLayerIn(self, next_layer, next_trg_names, fused_layer):
        tmp_list = []
        tmp_loc = []
        add_flag = True
        if next_layer:
            for idx, i in enumerate(next_layer.lparams['in']):
                if i[0] and i[0].name in next_trg_names:
                    if add_flag:
                        tmp_list.append([(fused_layer), i[1]])
                        tmp_loc.append(next_layer.getTileInfo().src_loc[idx])
                        add_flag = False
                else:
                    tmp_list.append([(i[0]), i[1]])
                    tmp_loc.append(next_layer.getTileInfo().src_loc[idx])

            next_layer.lparams['in'] = tmp_list
            next_layer.getTileInfo().src_loc = tmp_loc

    def findStructureByType(self, node, node_ref, node_list):
        compare_list = [(node, node_ref)]
        ref2node = {}
        node2ref = {}

        node_list_unsorted = []
        while len(compare_list) > 0:
            node, node_ref = compare_list.pop(0)
        
            if id(node_ref) in ref2node and ref2node[id(node_ref)] != node:
                return False
            elif id(node) in node2ref and node2ref[id(node)] != node_ref:
                return False

            if node in node_list_unsorted or node_ref.op_type == "":
                continue

            if node.op_type != node_ref.op_type:
                return False

            ref2node[id(node_ref)] = node
            node2ref[id(node)] = node_ref
            node_list_unsorted.append(node)

            node_prev = node.getPrevNodes()
            node_next = node.getNextNodes()
            node_ref_prev = node_ref.getPrevNodes()
            node_ref_next = node_ref.getNextNodes()

            if len(node_prev) != len(node_ref_prev) and len(node_ref_prev) > 0:
                return False

            if len(node_next) != len(node_ref_next) and len(node_ref_next) > 0:
                return False

            if len(node_next) > 0:
                compare_list.extend(list(zip(node_next, node_ref_next)))

            if len(node_prev) > 1 :
                compare_list.extend(list(zip(node_prev, node_ref_prev)))
        
        for node in self.graph_nodes:
            node_list.append(ref2node[id(node)])

        return True


class LayerNormReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"]
        self.graph_nodes = self.initNodeGraph()
        self.axis = -1
        self.eps = 1e-5

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])
        nodes[1].setPrevNodes([make_simple_node("", ""), nodes[0]])
        nodes[1].setNextNodes([nodes[2], nodes[6]])
        
        for idx in [2, 3, 4, 5, 7]:
            nodes[idx].setPrevNodes([nodes[idx-1]])
            nodes[idx].setNextNodes([nodes[idx+1]])

        nodes[6].setPrevNodes([nodes[1], nodes[5]])
        nodes[6].setNextNodes([nodes[7]])

        nodes[8].setPrevNodes([nodes[7]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        for node_reduce_mean in [node_list[0], node_list[3]]:
            in_shapes = node_reduce_mean.getInputShapes()
            out_shapes = node_reduce_mean.getOutputShapes()
            if len(in_shapes) != len(out_shapes) or len(in_shapes) != 1 or np.sum(in_shapes[0] != out_shapes[0]) > 1:
                return False

        self.axis = int(np.argmax(np.array(node_list[0].getInputShapes()[0]) != np.array(node_list[0].getOutputShapes()[0])))

        node_pow_weights = node_list[2].getInputWeights()
        if node_pow_weights is None or len(node_pow_weights) > 1 or node_pow_weights[0] != 2:
            return False

        node_add_weights = node_list[4].getInputWeights()
        if node_add_weights is None or len(node_add_weights) > 1 or np.prod(node_add_weights[0].shape) != 1:
            return False
        
        self.eps = float(node_add_weights[0])

        node_add_weights = node_list[-1].getInputWeights()
        node_mul_weights = node_list[-2].getInputWeights()
        weight_shape = node_list[0].getInputShapes()[0][self.axis:]
        if len(node_add_weights) != len(node_mul_weights) or np.prod(weight_shape) != np.prod(node_add_weights[0].shape) or np.prod(node_add_weights[0].shape) != np.prod(node_mul_weights[0].shape):
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):       
        inputs = [node_list[0].input[0], node_list[7].input[1], node_list[8].input[1]]
        outputs = node_list[-1].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_LayerNorm_ptool",
            op_type="LayerNormalization",
            axis=self.axis,
            epsilon=self.eps,
            inputs=inputs,
            outputs=outputs
            )
        return [NodeRef(new_node)]


class LayerNormTFReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["ReduceMean", "Sub", "Mul", "ReduceMean", "Add", "Sqrt", "Reciprocal", "Mul", "Mul", "Add"]
        self.graph_nodes = self.initNodeGraph()
        self.axis = -1
        self.eps = 1e-5

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])
        nodes[1].setPrevNodes([make_simple_node(), nodes[0]])
        nodes[1].setNextNodes([nodes[2], nodes[7]])
        
        nodes[2].setPrevNodes([nodes[1], nodes[1]])
        nodes[2].setNextNodes([nodes[3]])

        for idx in [3, 5, 6, 8]:
            nodes[idx].setPrevNodes([nodes[idx-1]])
            nodes[idx].setNextNodes([nodes[idx+1]])

        nodes[4].setPrevNodes([nodes[3]])
        nodes[4].setNextNodes([nodes[5]])

        nodes[7].setPrevNodes([nodes[1], nodes[6]])
        nodes[7].setNextNodes([nodes[8]])

        nodes[9].setPrevNodes([nodes[8]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        for node_reduce_mean in [node_list[0], node_list[3]]:
            in_shapes = node_reduce_mean.getInputShapes()
            out_shapes = node_reduce_mean.getOutputShapes()
            if len(in_shapes) != len(out_shapes) or len(in_shapes) != 1 or np.sum(in_shapes[0] != out_shapes[0]) > 1:
                return False

        self.axis = int(np.argmax(np.array(node_list[0].getInputShapes()[0]) != np.array(node_list[0].getOutputShapes()[0])))

        node_mul = node_list[2]
        if len(node_mul.input) != 2 or node_mul.input[0] != node_mul.input[1]:
            return False

        node_add_weights = node_list[4].getInputWeights()
        if node_add_weights is None or len(node_add_weights) > 1 or np.prod(node_add_weights[0].shape) != 1:
            return False
        
        self.eps = float(node_add_weights[0])

        node_add_weights = node_list[-1].getInputWeights()
        node_mul_weights = node_list[-2].getInputWeights()
        weight_shape = node_list[0].getInputShapes()[0][self.axis:]
        if len(node_add_weights) != len(node_mul_weights) or np.prod(weight_shape) != np.prod(node_add_weights[0].shape) or np.prod(node_add_weights[0].shape) != np.prod(node_mul_weights[0].shape):
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):      
        inputs = [node_list[0].input[0], node_list[7].input[1], node_list[8].input[1]]
        outputs = node_list[-1].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_LayerNorm_ptool",
            op_type="LayerNormalization",
            axis=self.axis,
            epsilon=self.eps,
            inputs=inputs,
            outputs=outputs
            )
        return [NodeRef(new_node)]


class GroupNormReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Reshape", "InstanceNormalization", "Reshape", "Mul", "Add"]
        self.graph_nodes = self.initNodeGraph()
        self.num_group = None
        self.eps = 1e-5

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])
        for idx in [1, 2, 3]:
            nodes[idx].setPrevNodes([nodes[idx-1]])
            nodes[idx].setNextNodes([nodes[idx+1]])

        nodes[4].setPrevNodes([nodes[3]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        initializer = kwargs["initializer"]
        reshape_shape = None
        scale = None
        bias = None

        # check first rehape rank 3
        if len(node_list[0].input) > 1 and len(node_list[2].input) > 1:
            for init in initializer:
                if init.name == node_list[0].input[1]:
                    reshape_shape = numpy_helper.to_array(init)
        else:
            return False
        
        if reshape_shape is None:
            return False
        if len(reshape_shape) != 3:
            return False

        self.num_group = reshape_shape[-2]

        # check instanceNorm scale=1, B=0
        for init in initializer:
            if init.name == node_list[1].input[1]:
                tmp = numpy_helper.to_array(init)

                if tmp != 1:
                    return False
            elif init.name == node_list[1].input[2]:
                tmp = numpy_helper.to_array(init)
                if tmp != 0:
                    return False

            elif init.name == node_list[-2].input[1]:
                scale = numpy_helper.to_array(init)
            
            elif init.name == node_list[-1].input[1]:
                bias = numpy_helper.to_array(init)

        # check mul & add same param shape
        if np.sum(scale.shape != bias.shape) > 0:
            return False
        
        for attr in node_list[1].attribute:
            if attr.name == "epsilon":
                self.eps = attr.f

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        inputs = [node_list[0].input[0], node_list[3].input[1], node_list[4].input[1]]
        outputs = node_list[-1].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_GroupNorm_ptool",
            op_type="GroupNormalization",
            num_group=self.num_group,
            epsilon=self.eps,
            inputs=inputs,
            outputs=outputs
            )
        return [NodeRef(new_node)]


class GeLUReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Div", "Erf", "Add", "Mul", "Mul"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])
        
        for idx in [1, 2]:
            nodes[idx].setPrevNodes([nodes[idx-1]])
            nodes[idx].setNextNodes([nodes[idx+1]])

        nodes[3].setPrevNodes([make_simple_node("", ""), nodes[2]])
        nodes[3].setNextNodes([nodes[4]])

        nodes[4].setPrevNodes([nodes[3]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        val_sqrt2 = None
        val_add = None
        val_mul = None
        eps = 1e-7

        initializer = kwargs["initializer"]
        for init in initializer:
            if init.name == node_list[0].input[1]:
                val_sqrt2 = numpy_helper.to_array(init)
            elif init.name == node_list[2].input[1]:
                val_add = numpy_helper.to_array(init)
            elif init.name == node_list[4].input[1]:
                val_mul = numpy_helper.to_array(init)

        if val_sqrt2 is None or val_sqrt2 - 1.4142135381698608 > eps:
            return False

        if val_add is None or val_add - 1 > eps:
            return False

        if val_mul is None or val_mul - 0.5 > eps:
            return False

        if node_list[0].getPrevNodes()[0] not in node_list[3].getPrevNodes():
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_GeLU_ptool", "GeLU", node_list[0].input[0:1], node_list[-1].output)]


class SoftmaxReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["ReduceMax", "Sub", "Exp", "ReduceSum", "Div"]
        self.graph_nodes = self.initNodeGraph()
        self.axis = -1

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([make_simple_node("", ""), nodes[0]])
        nodes[1].setNextNodes([nodes[2]])

        nodes[2].setPrevNodes([nodes[1]])
        nodes[2].setNextNodes([nodes[3], nodes[4]])

        nodes[3].setPrevNodes([nodes[2]])
        nodes[3].setNextNodes([nodes[4]])

        nodes[4].setPrevNodes([nodes[2], nodes[3]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        for node_reduce in [node_list[0], node_list[3]]:
            in_shapes = node_reduce.getInputShapes()
            out_shapes = node_reduce.getOutputShapes()

            if len(in_shapes) != len(out_shapes) or len(in_shapes) != 1 or np.sum(in_shapes != out_shapes) > 1:
                return False

        axis_reducemax = int(np.argmax(np.array(node_list[0].getInputShapes()[0]) != np.array(node_list[0].getOutputShapes()[0])))
        axis_reducemean = int(np.argmax(np.array(node_list[3].getInputShapes()[0]) != np.array(node_list[3].getOutputShapes()[0])))

        if axis_reducemax != axis_reducemean:
            return False

        self.axis = axis_reducemean

        if node_list[0].getPrevNodes()[0] != node_list[1].getPrevNodes()[0] and node_list[0].getPrevNodes()[0] != node_list[1].getPrevNodes()[1]:
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        inputs = [node_list[0].input[0]]
        outputs = node_list[-1].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_Softmax_ptool",
            op_type="Softmax",
            axis=self.axis,
            inputs=inputs,
            outputs=outputs
            )
        return [NodeRef(new_node)]


class SoftminReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Neg", "Softmax"]
        self.graph_nodes = self.initNodeGraph()
        self.axis = -1

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])
        nodes[1].setPrevNodes([nodes[0]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        for attr in node_list[1].attribute:
            if attr.name == "axis":
                self.axis = attr.i

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        inputs = node_list[0].input
        outputs = node_list[-1].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_Softmin_ptool",
            op_type="Softmin",
            axis=self.axis,
            inputs=inputs,
            outputs=outputs
            )
        return [NodeRef(new_node)]


class SwishReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Mul", "Sigmoid", "Mul"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([nodes[0]])
        nodes[1].setNextNodes([nodes[2]])

        nodes[2].setPrevNodes([make_simple_node("", ""), nodes[1]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        val_mul = None

        initializer = kwargs["initializer"]
        for init in initializer:
            if init.name == node_list[0].input[0]:
                val_mul = numpy_helper.to_array(init)

        if val_mul is None or np.prod(val_mul.shape) > 1:
            return False

        if node_list[0].getPrevNodes()[0] not in node_list[2].getPrevNodes():
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_Swish_ptool", "Swish", node_list[0].input, node_list[-1].output)]


class SiLUReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Sigmoid", "Mul"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([make_simple_node("", ""), nodes[0]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        if node_list[0].getPrevNodes()[0] not in node_list[1].getPrevNodes():
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_SiLU_ptool", "SiLU", node_list[0].input, node_list[-1].output)]


class MishReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Softplus", "Tanh", "Mul"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([nodes[0]])
        nodes[1].setNextNodes([nodes[2]])

        nodes[2].setPrevNodes([make_simple_node("", ""), nodes[2]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        if node_list[0].getPrevNodes()[0] not in node_list[2].getPrevNodes():
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_Mish_ptool", "Mish", node_list[0].input, node_list[-1].output)]


class ChannelShuffleReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Reshape", "Transpose", "Reshape"]
        self.graph_nodes = self.initNodeGraph()
        self.num_group = None

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([nodes[0]])
        nodes[1].setNextNodes([nodes[2]])

        nodes[2].setPrevNodes([nodes[1]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        initializer = kwargs["initializer"]
        shape_0 = None
        shape_1 = None
        perm = None

        for init in initializer:
            if init.name == node_list[0].input[1]:
                shape_0 = numpy_helper.to_array(init)
            elif init.name == node_list[2].input[1]:
                shape_1 = numpy_helper.to_array(init)

        if shape_0 is None or shape_1 is None:
            return False

        if np.prod(shape_0.shape) != 5 or np.prod(shape_1.shape) != 4:
            return False

        self.num_group = shape_0[1]

        for attr in node_list[1].attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
                break

        if np.sum(np.array(perm) != np.array([0, 2, 1, 3, 4])) > 0:
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        inputs = [node_list[0].input[0]]
        outputs = node_list[2].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_ChannelShuffle_ptool",
            op_type="ChannelShuffle",
            num_group=self.num_group,
            inputs=inputs,
            outputs=outputs
            )
        return [NodeRef(new_node)]


class BNLLReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Exp", "Add", "Log"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([nodes[0]])
        nodes[1].setNextNodes([nodes[2]])

        nodes[2].setPrevNodes([nodes[1]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        initializer = kwargs["initializer"]
        val_add = None

        for init in initializer:
            if init.name == node_list[1].input[0]:
                val_add = numpy_helper.to_array(init)
                break
        
        if val_add is None or np.prod(val_add.shape) != 1 or val_add != 1:
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_BNLL_ptool", "BNLL", node_list[0].input, node_list[-1].output)]


class LogSigmoidReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Sigmoid", "Log"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([nodes[0]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_LogSigmoid_ptool", "LogSigmoid", node_list[0].input, node_list[-1].output)]


class TanhShrinkReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Tanh", "Sub"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([make_simple_node("", ""), nodes[0]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        if node_list[0].getPrevNodes()[0] not in node_list[1].getPrevNodes():
            return False

        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        return [make_simple_node(node_list[0].name + "_TanhShrink_ptool", "TanhShrink", node_list[0].input, node_list[-1].output)]


class UpsampleReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["Upsample"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        roi = numpy_helper.from_array(np.empty([0], dtype=np.float32), node_list[0].name + "_roi")
        kwargs["initializer"].append(roi)

        roi_value_info = helper.make_tensor_value_info(node_list[0].name + "_roi", onnx.TensorProto.FLOAT, [0])
        kwargs["value_info"].append(roi_value_info)

        inputs = [node_list[0].name.input[0], node_list[0].name + "_roi", node_list[0].name.input[1]]
        mode_string = ''
        for attr in node_list[0].attribute:
            if attr.name == 'mode':
                mode_string = attr.s

        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_Resize_ptool",
            op_type="Resize",
            coordinate_transformation_mode="asymmetric",
            cubic_coeff_a=-0.75,
            mode=mode_string,
            nearest_mode="floor",
            inputs=inputs,
            outputs=node_list[0].output
        )
        return [NodeRef(new_node)]


class MatMulReplace(GraphReplace):
    def __init__(self):
        self.graph_node_types = ["MatMul", "Add"]
        self.graph_nodes = self.initNodeGraph()

    def initNodeGraph(self):
        nodes = []
        for idx, op_type in enumerate(self.graph_node_types):
            nodes.append(make_simple_node("layer_{}".format(idx), op_type))

        nodes[0].setNextNodes([nodes[1]])

        nodes[1].setPrevNodes([nodes[0]])

        return nodes

    def chkLayersByAttr(self, node_list, kwargs):
        initializer = kwargs["initializer"]
        val_weight = None
        val_bias = None

        for init in initializer:
            if init.name == node_list[0].input[1]:
                val_weight = numpy_helper.to_array(init)
            elif init.name in [node_list[1].input[0], node_list[1].input[1]]:
                if val_bias is not None:
                    return False
                val_bias = numpy_helper.to_array(init)

        if val_weight is None or val_bias is None:
            return False

        if len(val_weight.shape) != 2:
            return False

        if len(val_bias.shape) != 1 or np.prod(val_bias.shape) != val_weight.shape[1]:
            return False
            
        return True

    def modifyValueInfo(self, node_list, kwargs):
        pass

    def replaceLayers(self, node_list, kwargs):
        inputs = [node_list[0].input[0], node_list[0].input[1], node_list[1].input[1]]
        outputs = node_list[1].output
        new_node = onnx.helper.make_node(
            name=node_list[0].name + "_Gemm_ptool",
            op_type="Gemm",
            transA=0,
            transB=0,
            inputs=inputs,
            outputs=outputs
        )
        return [NodeRef(new_node)]


def make_simple_node(name="", op_type="", inputs=[], outputs=[]):
    return NodeRef(helper.make_node(name=name, op_type=op_type, inputs=inputs, outputs=outputs))


def get_onnx_name2shape(model):
    weight_info = {}
    tensor_info = {}
    for v in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(v)
        weight_info[v.name] = arr
        tensor_info[v.name] = arr.shape
    
    for v in model.graph.input:
        tensor_info[v.name] = [dd.dim_value for dd in v.type.tensor_type.shape.dim]

    for v in model.graph.output:
        tensor_info[v.name] = [dd.dim_value for dd in v.type.tensor_type.shape.dim]

    for v in model.graph.value_info:
        tensor_info[v.name] = [dd.dim_value for dd in v.type.tensor_type.shape.dim]      

    return tensor_info, weight_info


def create_relation_graph(model):
    node_list = []
    node_graph_list = []

    name2node = {}
    top_name2node = {}
    bot_name2node = {}

    tensor_info, weight_info = get_onnx_name2shape(model)

    for node in model.graph.node:
        node_list.append(NodeRef(node))

    for node in node_list:
        assert node.name not in name2node, "name: {} duplicated".format(node.name)

        name2node[node.name] = node
        for inp in node.input:
            if inp not in bot_name2node:
                bot_name2node[inp] = []

            if node not in bot_name2node[inp]:
                bot_name2node[inp].append(node)
        
        for out in node.output:
            if out not in top_name2node:
                top_name2node[out] = []

            if node not in top_name2node[out]:
                top_name2node[out].append(node)

    for node in node_list:
        prev_list = []
        next_list = []
        for inp in node.input:
            if inp in top_name2node:
                prev_list.extend(top_name2node[inp])

        for out in node.output:
            if out in bot_name2node:
                next_list.extend(bot_name2node[out])

        # set prev/next node
        node.setPrevNodes(prev_list)
        node.setNextNodes(next_list)

        # set in/out shape
        in_shapes = []
        out_shapes = []
        for inp in node.input:
            if inp == "":
                continue
            assert inp in tensor_info, "in_name: {}".format(inp)
            in_shapes.append(tensor_info[inp])

        for out in node.output:
            assert out in tensor_info, "out_name: {}".format(out)
            out_shapes.append(tensor_info[out])

        node.setInputShapes(in_shapes)
        node.setOutputShapes(out_shapes)

        # set in weight
        in_weights = []
        for inp in node.input:
            if inp in weight_info:
                in_weights.append(weight_info[inp])

        node.setInputWeights(in_weights)

        node_graph_list.append(node)

    return node_graph_list


def remove_unused_infos(node_list, initialier_list, value_info_list):
    # remove unused initializer, value_info
    unused_init_list = []
    unused_vinfo_list = []
    buf_list = []
    for node in node_list:
        for inp in node.input:
            buf_list.append(inp)
        for out in node.output:
            buf_list.append(out)

    for v in initialier_list:
        if v.name not in buf_list:
            unused_init_list.append(v)

    for v in value_info_list:
        if v.name not in buf_list:
            unused_vinfo_list.append(v)

    for v in unused_init_list:
        initialier_list.remove(v)

    for v in unused_vinfo_list:
        value_info_list.remove(v)


def replace_onnx_op(model, pattern_replacer=None):
    fused_nodes = []

    if pattern_replacer is None:
        pattern_replacer = {}

        # standard onnx op
        pattern_replacer["ReduceMax"] = [SoftmaxReplace()]
        pattern_replacer["Upsample"] = [UpsampleReplace()]

        # standard onnx op from opset 17
        pattern_replacer["ReduceMean"] = [LayerNormReplace()]

        # standard onnx op from opset 18
        pattern_replacer["Reshape"] = [GroupNormReplace()]
        pattern_replacer["Softplus"] = [MishReplace()]

        # novatek defined op
        # pattern_replacer["MatMul"] = [MatMulReplace()]
        pattern_replacer["Neg"] = [SoftminReplace()]
        pattern_replacer["Div"] = [GeLUReplace()]
        pattern_replacer["Sigmoid"] = [SiLUReplace(), LogSigmoidReplace()]
        pattern_replacer["Mul"] = [SwishReplace()]
        pattern_replacer["Reshape"].append(ChannelShuffleReplace())
        pattern_replacer["Exp"] = [BNLLReplace()]
        pattern_replacer["Tanh"] = [TanhShrinkReplace()]

    assert type(pattern_replacer) == dict

    kwargs = {}
    kwargs["input"] = model.graph.input
    kwargs["output"] = model.graph.output
    kwargs["initializer"] = model.graph.initializer
    kwargs["value_info"] = model.graph.value_info
    kwargs["doc_string"] = model.graph.doc_string
    kwargs["sparse_initializer"] = model.graph.sparse_initializer

    checked_list = []

    node_rel_list = create_relation_graph(model)

    for node in node_rel_list:
        name = node.name
        if node.name not in checked_list:
            if node.op_type in pattern_replacer:
                for replacer in pattern_replacer[node.op_type]:
                    out_node_list = []
                    replaced_list = []
                    is_success = replacer(node, out_node_list, replaced_list, kwargs=kwargs)

                    if is_success:
                        checked_list.extend([l.name for l in replaced_list])
                        for fused_node in out_node_list:
                            fused_nodes.append(fused_node)
                        
                        break
                else:
                    fused_nodes.append(node)
                    checked_list.append(name)

            else:
                fused_nodes.append(node)
                checked_list.append(name)

    remove_unused_infos(fused_nodes, kwargs["initializer"], kwargs["value_info"])

    fused_nodes = [f.onnx_node for f in fused_nodes]
    graph_def = helper.make_graph(fused_nodes,
        name=model.graph.name,
        inputs=kwargs["input"], 
        outputs=kwargs["output"],
        initializer=kwargs["initializer"],
        value_info=kwargs["value_info"],
        doc_string=kwargs["doc_string"],
        sparse_initializer=kwargs["sparse_initializer"],)

    model.graph.CopyFrom(graph_def)

    return model


def simplify_onnx_graph(model):
    model = replace_onnx_op(model)
    return model