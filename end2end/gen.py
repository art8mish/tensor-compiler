import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
ONNX_MODELS_PATH = PROJECT_PATH / "models"
ONNX_MODEL_PATH = ONNX_MODELS_PATH / "model.onnx"


def create_lenet_like_onnx(path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    def randn(shape):
        # Smaller variance keeps activations in a numerically stable range and
        # reduces backend-dependent drift in deep conv reductions.
        return (0.1 * rng.standard_normal(shape)).astype(np.float32)

    nodes = []
    initializers = []

    def add_initializer(name: str, arr: np.ndarray) -> None:
        initializers.append(
            helper.make_tensor(name, TensorProto.FLOAT, list(arr.shape), arr.flatten())
        )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 28, 28])
    prev = "X"

    # 1) C1: Conv 1->6, 5x5 (1x1x28x28 -> 1x6x28x28)
    W0 = randn((6, 1, 5, 5))
    B0 = randn((6,))
    add_initializer("W0", W0)
    add_initializer("B0", B0)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W0", "B0"],
            outputs=["n_conv1"],
            name="conv1",
            kernel_shape=[5, 5],
            strides=[1, 1],
            pads=[2, 2, 2, 2],
            dilations=[1, 1],
            group=1,
        )
    )
    prev = "n_conv1"
    nodes.append(helper.make_node("Relu", inputs=[prev], outputs=["n_relu1"], name="relu1"))
    prev = "n_relu1"

    # 2) S2: subsampling, Conv 2x2 (1x6x28x28 -> 1x6x14x14)
    W_s2 = randn((6, 6, 2, 2))
    B_s2 = randn((6,))
    add_initializer("W_s2", W_s2)
    add_initializer("B_s2", B_s2)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W_s2", "B_s2"],
            outputs=["n_s2"],
            name="s2_conv",
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    )
    nodes.append(helper.make_node("Relu", inputs=["n_s2"], outputs=["n_relu2"], name="relu2"))
    prev = "n_relu2"

    # 3) C3: Conv 6->16, 5x5 (1x6x14x14 -> 1x16x10x10)
    W_c3 = randn((16, 6, 5, 5))
    B_c3 = randn((16,))
    add_initializer("W_c3", W_c3)
    add_initializer("B_c3", B_c3)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W_c3", "B_c3"],
            outputs=["n_c3"],
            name="c3_conv",
            kernel_shape=[5, 5],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    )
    prev = "n_c3"
    nodes.append(helper.make_node("Relu", inputs=[prev], outputs=["n_relu3"], name="relu3"))
    prev = "n_relu3"

    # 4) S4: subsampling, Conv 2x2 (1x16x10x10 -> 1x16x5x5)
    W_s4 = randn((16, 16, 2, 2))
    B_s4 = randn((16,))
    add_initializer("W_s4", W_s4)
    add_initializer("B_s4", B_s4)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W_s4", "B_s4"],
            outputs=["n_s4"],
            name="s4_conv",
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    )
    nodes.append(helper.make_node("Relu", inputs=["n_s4"], outputs=["n_relu4"], name="relu4"))
    prev = "n_relu4"

    # 5) C5: Conv 16->120, 5x5 (1x16x5x5 -> 1x120x1x1)
    W_c5 = randn((120, 16, 5, 5))
    B_c5 = randn((120,))
    add_initializer("W_c5", W_c5)
    add_initializer("B_c5", B_c5)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W_c5", "B_c5"],
            outputs=["n_c5"],
            name="c5_conv",
            kernel_shape=[5, 5],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    )
    prev = "n_c5"
    nodes.append(helper.make_node("Relu", inputs=[prev], outputs=["n_relu5"], name="relu5"))
    prev = "n_relu5"

    # 6) F6: Conv 1x1 (1x120x1x1 -> 1x84x1x1)
    W_f6 = randn((84, 120, 1, 1))
    B_f6 = randn((84,))
    add_initializer("W_f6", W_f6)
    add_initializer("B_f6", B_f6)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W_f6", "B_f6"],
            outputs=["n_f6"],
            name="f6_conv1x1",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    )
    prev = "n_f6"
    nodes.append(helper.make_node("Relu", inputs=[prev], outputs=["n_relu6"], name="relu6"))
    prev = "n_relu6"

    # 7) Output: Conv 1x1 (1x84x1x1 -> 1x10x1x1)
    W_out = randn((10, 84, 1, 1))
    B_out = randn((10,))
    add_initializer("W_out", W_out)
    add_initializer("B_out", B_out)
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[prev, "W_out", "B_out"],
            outputs=["Y"],
            name="conv_out",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    )

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10, 1, 1])

    graph_def = helper.make_graph(nodes, "LeNetClassicLike", [X], [Y], initializer=initializers)
    model_def = helper.make_model(graph_def, producer_name="tensor-compiler-e2e")
    model_def.opset_import[0].version = 13

    onnx.checker.check_model(model_def)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_def, path)
    print(f"Model saved: {path}")


if __name__ == "__main__":
    create_lenet_like_onnx(ONNX_MODEL_PATH)
