import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np
from pathlib import Path

def create_test_onnx(file_name="test_model.onnx"):
    # shape for Conv: [Batch, Channels, Height, Width] -> [1, 1, 5, 5]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 5, 5])
    
    # Initializers
    # weights for Conv: [Out_Channels, In_Channels, kH, kW] -> [1, 1, 3, 3]
    conv_w_data = np.random.randn(1, 1, 3, 3).astype(np.float32)
    conv_w = helper.make_tensor('W', TensorProto.FLOAT, [1, 1, 3, 3], conv_w_data.flatten())
    
    # Conv [1,1,5,5] -> [1,1,3,3]
    # Gemm
    gemm_a = helper.make_tensor_value_info('gemm_a', TensorProto.FLOAT, [1, 10])
    gemm_b_data = np.random.randn(10, 5).astype(np.float32)
    gemm_b = helper.make_tensor('gemm_b', TensorProto.FLOAT, [10, 5], gemm_b_data.flatten())
    
    # Conv
    node_conv = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['conv_out'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
        name='test_conv'
    )
    
    # Relu
    node_relu = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['relu_out'],
        name='test_relu'
    )
    
    # Add (const)
    bias_data = np.array([1.0], dtype=np.float32)
    bias_tensor = helper.make_tensor('add_bias', TensorProto.FLOAT, [1], bias_data)
    
    node_add = helper.make_node(
        'Add',
        inputs=['relu_out', 'add_bias'],
        outputs=['final_out'],
        name='test_add'
    )

    # graph
    graph_def = helper.make_graph(
        [node_conv, node_relu, node_add],
        'TestGraph',
        [X],
        [helper.make_tensor_value_info('final_out', TensorProto.FLOAT, [1, 1, 3, 3])],
        [conv_w, bias_tensor]
    )

    model_def = helper.make_model(graph_def, producer_name='onnx-test-gen')
    model_def.opset_import[0].version = 13

    onnx.checker.check_model(model_def)
    onnx.save(model_def, file_name)
    print(f"Model is save: {file_name}")

if __name__ == "__main__":
    test_model_path = Path(__file__).parent / "models" / "test_model.onnx"
    test_model_path.parent.mkdir(parents=True, exist_ok=True)
    create_test_onnx(test_model_path)