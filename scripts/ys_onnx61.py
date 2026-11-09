"""ys_onnx61.py <in73.onnx> <out61.onnx>: 7-13 check 2. Wrap baseline v3's exported 73-D policy (61 + J's camera tracker 12,
normalizer inside) as a 61-D policy for the runtime's contract: a Concat in front feeds the 12 tracker inputs zeros ("never
seen"), everything else (normalizer, MLP, metadata) unchanged. Checks the graph and that out61(x) == out73([x, 0])."""
import sys
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
old = g.input[0]; assert old.name == "obs" and old.type.tensor_type.shape.dim[1].dim_value == 73
for nd in g.node:  # the original graph now reads "obs73"
  nd.input[:] = ["obs73" if i == "obs" else i for i in nd.input]
g.initializer.append(numpy_helper.from_array(np.zeros((1, 12), np.float32), "cam_none"))
g.node.insert(0, helper.make_node("Concat", ["obs", "cam_none"], ["obs73"], axis=1, name="pad_camera_tracker"))
g.input.remove(old); g.input.insert(0, helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61]))
meta = {p.key: p.value for p in m.metadata_props}
meta["wrapper"] = "61-D: actor obs 61:73 (J camera tracker: leader head + red ball, cur/prev u v valid) fed zeros (scripts/ys_onnx61.py)"
del m.metadata_props[:]
for k, v in meta.items(): m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
onnx.checker.check_model(m); onnx.save(m, dst)
a, b = ort.InferenceSession(src), ort.InferenceSession(dst); rng = np.random.default_rng(0); d = 0.0
for _ in range(200):
  x = rng.normal(size=(1, 61)).astype(np.float32)
  d = max(d, float(np.abs(a.run(None, {"obs": np.concatenate([x, np.zeros((1, 12), np.float32)], 1)})[0] - b.run(None, {"obs": x})[0]).max()))
print("saved", dst, "inputs", [(i.name, i.shape) for i in b.get_inputs()], "max |out61 - out73(pad 0)| over 200 random obs:", d)
