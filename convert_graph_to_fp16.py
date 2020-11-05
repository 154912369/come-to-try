convert_graph_to_fp16(model_path, save_path, name, as_text=as_text, target_type=target_type, input_name=input_name, output_names=output_names)import tensorflow as tf
from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format
import numpy as np

# object detection api input and output nodes
input_name = ["Input-Token","Input-Segment"]
output_names = ["logits/Softmax"]
# Const should be float32 in object detection api during nms (see here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/non-max-suppression-v4.html)
keep_fp32_node_name = ["Postprocessor/BatchMultiClassNonMaxSuppression/MultiClassNonMaxSuppression/non_max_suppression/iou_threshold",
                       "Postprocessor/BatchMultiClassNonMaxSuppression/MultiClassNonMaxSuppression/non_max_suppression/score_threshold"]

def load_graph(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        if model_path.endswith("pb"):
            with open(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(model_path, "r") as pf:
                text_format.Parse(pf.read(), graph_def)
        tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)
        return sess

def rewrite_batch_norm_node_v2(node, graph_def, target_type='fp16'):
    """
    Rewrite FusedBatchNorm with FusedBatchNormV2 for reserve_space_1 and reserve_space_2 in FusedBatchNorm require float32 for
    gradient calculation (See here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm)
    """
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT
    new_node = graph_def.node.add()
    new_node.op = "FusedBatchNormV2"
    new_node.name = node.name
    new_node.input.extend(node.input)
    new_node.attr["U"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
    for attr in list(node.attr.keys()):
        if attr == "T":
            node.attr[attr].type = dtype
    print("rewrite fused_batch_norm done!")



def convert_graph_to_fp16(model_path, save_path, name, as_text=False, target_type='fp16', input_name=None, output_names=None):
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT
    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(source_graph_def.versions)

    for node in source_graph_def.node:

        # fused batch norm node
        if node.op == "FusedBatchNorm":
            rewrite_batch_norm_node_v2(node, target_graph_def, target_type=target_type)
            continue
        # replicate node
        new_node = target_graph_def.node.add()
        new_node.op = node.op
        new_node.name = node.name

        new_node.input.extend(node.input)
        attrs = list(node.attr.keys())

        if ("BatchNorm" in node.name) or ('batch_normalization' in node.name):
            for attr in attrs:
                new_node.attr[attr].CopyFrom(node.attr[attr])
            continue

        for attr in attrs:
            if node.attr[attr].type == types_pb2.DT_FLOAT :
                # modify node dtype
                new_node.attr[attr].type = dtype

            if attr == "value" or attr in ["dtype","SrcT","T","Tparams","DstT"]:
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_FLOAT:
                    # if float_val exists
                    if tensor.float_val:
                        #print(node_name)
                        float_val = tf.make_ndarray(node.attr[attr].tensor)
                        float_val[float_val>tf.float16.max]=tf.float16.max
                        float_val[float_val<tf.float16.min]=tf.float16.min

                        new_node.attr[attr].tensor.CopyFrom(tf.make_tensor_proto(float_val, dtype=dtype))
                        if new_node.attr[attr].tensor.dtype:
                            new_node.attr[attr].tensor.dtype=dtype
                        if new_node.attr[attr].type:
                            new_node.attr[attr].type = dtype
                        continue
                    # if tensor content exists
                    if tensor.tensor_content:
                        #print(node.name)
                        tensor_shape = [x.size for x in tensor.tensor_shape.dim]
                        tensor_weights = tf.make_ndarray(tensor)
                        tensor_weights[tensor_weights > tf.float16.max] = tf.float16.max
                        tensor_weights[tensor_weights < tf.float16.min] = tf.float16.min
                        # reshape tensor
                        tensor_weights = np.reshape(tensor_weights, tensor_shape)
                        tensor_proto = tf.make_tensor_proto(tensor_weights, dtype=dtype)
                        new_node.attr[attr].tensor.CopyFrom(tensor_proto)
                        if new_node.attr[attr].tensor.dtype:
                            new_node.attr[attr].tensor.dtype = dtype
                        if new_node.attr[attr].type:
                            new_node.attr[attr].type = dtype
                        continue
            new_node.attr[attr].CopyFrom(node.attr[attr])
            if new_node.attr[attr].type == types_pb2.DT_FLOAT  and (attr in ["dtype","SrcT","T","Tparams","DstT"] ):
                # modify node dtype
                new_node.attr[attr].type = dtype
    # transform graph
    if output_names:
        if not input_name:
            input_name = []
        transforms = ["strip_unused_nodes"]
        target_graph_def = TransformGraph(target_graph_def, input_name, output_names, transforms)
    # write graph_def to model
    tf.io.write_graph(target_graph_def, logdir=save_path, name=name, as_text=as_text)
    print("Converting done ...")

save_path = "/Users/renweijie/code/python/bert4keras/data/pb_file"
name = "test.pb"
as_text = False
target_type = 'fp16'
model_path="/Users/renweijie/code/python/bert4keras/data/pb_file/model.pb"
convert_graph_to_fp16(model_path, save_path, name, as_text=as_text, target_type=target_type, input_name=input_name, output_names=output_names)
