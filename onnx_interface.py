import onnx
import onnxruntime
import pickle
import os
import numpy as np
import torch

def load(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model


class OnnxInterface:
    def __init__(self,path):
        self.device = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        self.model = load(path)
        self.sess = onnxruntime.InferenceSession(path,providers=[self.device])

    def predict(self,sample):
        x_input = sample.astype(np.float32)[np.newaxis, ...]
        x_input = np.transpose(x_input, (0, 2, 1))
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        res = self.sess.run([output_name], {input_name: x_input})
        output_atan = np.arctan2(res[0][0][0], res[0][0][1]) / (2 * np.pi)
        out = ((output_atan + 1) % 1.0) * 100
        return out



if __name__ == "__main__":
    mInterface = OnnxInterface(r"models/model_3.onnx")
    path = r"Y:\Datasets\Fyzio"
    exercise = "Wide squat"

    with open(os.path.join(path, "X_train", exercise + ".pkl"), "rb") as f:
        x = np.array(pickle.load(f))
        print(x.shape)
    for i in range(1000):
        print(mInterface.predict(x[i]))