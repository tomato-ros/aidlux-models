from __future__ import annotations

import time
from typing import List, Sequence

import aidlite
import numpy as np

from utils import detect_postprocess_outputs, preprocess_image


class YoloModel:
    def __init__(
            self,
            model_path: str,
            output_shapes: Sequence[Sequence[int]],
            output_layout: str,
            width: int = 640,
            height: int = 640,
            class_num: int = 80,
            model_type: str = "qnn",
    ):
        self.width = width
        self.height = height
        self.class_num = class_num
        self.output_shapes = [list(shape) for shape in output_shapes]
        self.output_layout = output_layout

        model = aidlite.Model.create_instance(model_path)
        if model is None:
            raise RuntimeError("Create model failed")
        
        model.set_model_properties(
            [[1, height, width, 3]],
            aidlite.DataType.TYPE_FLOAT32,
            self.output_shapes,
            aidlite.DataType.TYPE_FLOAT32,
        )

        config = aidlite.Config.create_instance()
        if config is None:
            raise RuntimeError("Create config failed")
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        if model_type.lower() == "qnn":
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        elif model_type.lower() in ("snpe", "snpe2"):
            config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if interpreter is None:
            raise RuntimeError("Build interpreter failed")
        if interpreter.init() != 0:
            raise RuntimeError("Interpreter init failed")
        if interpreter.load_model() != 0:
            raise RuntimeError("Interpreter load model failed")
        self.interpreter = interpreter

    def __del__(self):
        interpreter = getattr(self, "interpreter", None)
        if interpreter is not None:
            interpreter.destory()

    def __call__(
            self,
            image: np.ndarray,
            invoke_nums: int = 10,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
    ) -> np.ndarray:
        input_tensor, scale = preprocess_image(image, self.width)
        if self.interpreter.set_input_tensor(0, input_tensor.data) != 0:
            raise RuntimeError("set_input_tensor failed")

        invoke_time = []
        for _ in range(invoke_nums):
            t1 = time.time()
            result = self.interpreter.invoke()
            invoke_time.append((time.time() - t1) * 1000)
            if result != 0:
                raise RuntimeError("interpreter invoke failed")

        print('\033[1;32m----------------------------------------\033[0m')
        print(
            f"QNN invoke {invoke_nums} times:\n"
            f" --mean_invoke_time is {sum(invoke_time) / invoke_nums}\n"
            f" --max_invoke_time is {max(invoke_time)}\n"
            f" --min_invoke_time is {min(invoke_time)}\n"
            f" --var_invoketime is {np.var(invoke_time)}"
        )

        print('\033[1;32m----------------------------------------\033[0m')

        outputs = [
            self.interpreter.get_output_tensor(i).reshape(*shape)
            for i, shape in enumerate(self.output_shapes)
        ]

        return detect_postprocess_outputs(
            outputs,
            self.output_layout,
            image.shape,
            scale,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            class_num=self.class_num,
        )
