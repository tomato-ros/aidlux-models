import time

import aidlite
import numpy as np


class AidliteTool:

    def load_encoder(self, model_path):

        print(model_path)

        config = aidlite.Config.create_instance()
        if config is None:
            print("Create config failed !")
            return False

        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.framework_type = aidlite.FrameworkType.TYPE_QNN248
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        model = aidlite.Model.create_instance(model_path)

        self.input_shapes = [[1, 256], [1, 256]]
        self.output_shapes = [[1, 8, 256, 64], [1, 8, 256, 64], [1, 8, 256, 64], [1, 8, 256, 64], [1, 8, 256, 64],
                              [1, 8, 256, 64], [1, 8, 256, 64], [1, 8, 256, 64], [1, 8, 256, 64], [1, 8, 256, 64],
                              [1, 8, 256, 64], [1, 8, 256, 64]]

        model.set_model_properties(self.input_shapes, aidlite.DataType.TYPE_FLOAT32,
                                   self.output_shapes, aidlite.DataType.TYPE_FLOAT32)

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)

        if self.interpreter is None:
            print("build_interpretper_from_model_and_config failed !")
            return None
        result = self.interpreter.init()
        if result != 0:
            print(f"interpreter init failed !")
            return False
        result = self.interpreter.load_model()
        if result != 0:
            print("interpreter load model failed !")
            return False
        return self.interpreter

    def load_decoder(self, model_path):
        print(model_path)

        config = aidlite.Config.create_instance()

        if config is None:
            print("Create config failed !")
            return False

        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.framework_type = aidlite.FrameworkType.TYPE_QNN248
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        model = aidlite.Model.create_instance(model_path)

        self.input_shapes = [[1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64],
                             [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64],
                             [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64],
                             [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64],
                             [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64],
                             [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64], [1, 8, 255, 64],
                             [1, 256], [1, 1], [1], ]

        self.output_shapes = [[1, 8, 1, 64], [1, 8, 1, 64],
                              [1, 8, 1, 64], [1, 8, 1, 64],
                              [1, 8, 1, 64], [1, 8, 1, 64],
                              [1, 8, 1, 64], [1, 8, 1, 64],
                              [1, 8, 1, 64], [1, 8, 1, 64],
                              [1, 8, 1, 64], [1, 8, 1, 64],
                              [1, 1, 65001]]

        model.set_model_properties(self.input_shapes, aidlite.DataType.TYPE_FLOAT32,
                                   self.output_shapes, aidlite.DataType.TYPE_FLOAT32)

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)

        if self.interpreter is None:
            print("build_interpretper_from_model_and_config failed !")
            return None
        result = self.interpreter.init()

        if result != 0:
            print(f"interpreter init failed !")
            return False

        result = self.interpreter.load_model()

        if result != 0:
            print("interpreter load model failed !")
            return False

        return self.interpreter

    def init(self, encoder_model_path, decoder_model_path):
        self.encoder = self.load_encoder(encoder_model_path)
        self.decoder = self.load_decoder(decoder_model_path)

    def decoder_infer(self, data):
        result = self.decoder.set_input_tensor(0, data["input_ids"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(1, data["encoder_attention_mask"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(2, data["position"].astype(np.float32).data)

        result = self.decoder.set_input_tensor(3, data["block_0_past_self_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(4, data["block_0_past_self_value_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(5, data["block_0_cross_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(6, data["block_0_cross_value_states"].astype(np.float32).data)

        result = self.decoder.set_input_tensor(7, data["block_1_past_self_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(8, data["block_1_past_self_value_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(9, data["block_1_cross_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(10, data["block_1_cross_value_states"].astype(np.float32).data)

        result = self.decoder.set_input_tensor(11, data["block_2_past_self_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(12, data["block_2_past_self_value_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(13, data["block_2_cross_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(14, data["block_2_cross_value_states"].astype(np.float32).data)

        result = self.decoder.set_input_tensor(15, data["block_3_past_self_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(16, data["block_3_past_self_value_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(17, data["block_3_cross_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(18, data["block_3_cross_value_states"].astype(np.float32).data)

        result = self.decoder.set_input_tensor(19, data["block_4_past_self_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(20, data["block_4_past_self_value_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(21, data["block_4_cross_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(22, data["block_4_cross_value_states"].astype(np.float32).data)

        result = self.decoder.set_input_tensor(23, data["block_5_past_self_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(24, data["block_5_past_self_value_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(25, data["block_5_cross_key_states"].astype(np.float32).data)
        result = self.decoder.set_input_tensor(26, data["block_5_cross_value_states"].astype(np.float32).data)

        for i in range(1):
            begin_time = time.time()
            self.decoder.invoke()
            # print("解码时间:{0}".format(time.time() - begin_time))

        res_list = []

        logits = self.decoder.get_output_tensor(12).reshape(1, 1, 65001)
        res_list.append(logits)

        block_0_present_self_key_states = self.decoder.get_output_tensor(0).reshape(1, 8, 1, 64)
        block_0_present_self_value_states = self.decoder.get_output_tensor(1).reshape(1, 8, 1, 64)
        res_list.append(block_0_present_self_key_states)
        res_list.append(block_0_present_self_value_states)

        block_1_present_self_key_states = self.decoder.get_output_tensor(2).reshape(1, 8, 1, 64)
        block_1_present_self_value_states = self.decoder.get_output_tensor(3).reshape(1, 8, 1, 64)
        res_list.append(block_1_present_self_key_states)
        res_list.append(block_1_present_self_value_states)

        block_2_present_self_key_states = self.decoder.get_output_tensor(4).reshape(1, 8, 1, 64)
        block_2_present_self_value_states = self.decoder.get_output_tensor(5).reshape(1, 8, 1, 64)
        res_list.append(block_2_present_self_key_states)
        res_list.append(block_2_present_self_value_states)

        block_3_present_self_key_states = self.decoder.get_output_tensor(6).reshape(1, 8, 1, 64)
        block_3_present_self_value_states = self.decoder.get_output_tensor(7).reshape(1, 8, 1, 64)
        res_list.append(block_3_present_self_key_states)
        res_list.append(block_3_present_self_value_states)

        block_4_present_self_key_states = self.decoder.get_output_tensor(8).reshape(1, 8, 1, 64)
        block_4_present_self_value_states = self.decoder.get_output_tensor(9).reshape(1, 8, 1, 64)
        res_list.append(block_4_present_self_key_states)
        res_list.append(block_4_present_self_value_states)

        block_5_present_self_key_states = self.decoder.get_output_tensor(10).reshape(1, 8, 1, 64)
        block_5_present_self_value_states = self.decoder.get_output_tensor(11).reshape(1, 8, 1, 64)
        res_list.append(block_5_present_self_key_states)
        res_list.append(block_5_present_self_value_states)

        return res_list

    def encoder_infer(self, data):
        # pdb.set_trace()
        result = self.encoder.set_input_tensor(0, data["input_ids"].astype(np.float32).data)
        result = self.encoder.set_input_tensor(1, data["encoder_attention_mask"].astype(np.float32).data)

        for i in range(1):
            begin_time = time.time()

            self.encoder.invoke()

            # print('\033[1;32m################################################################################\033[0m')
            # print("编码时间:", time.time() - begin_time)

        res_list = []
        block_0_cross_key_states = self.encoder.get_output_tensor(0).reshape(1, 8, 256, 64)
        block_0_cross_value_states = self.encoder.get_output_tensor(1).reshape(1, 8, 256, 64)
        res_list.append(block_0_cross_key_states)
        res_list.append(block_0_cross_value_states)

        block_1_cross_key_states = self.encoder.get_output_tensor(2).reshape(1, 8, 256, 64)
        block_1_cross_value_states = self.encoder.get_output_tensor(3).reshape(1, 8, 256, 64)
        res_list.append(block_1_cross_key_states)
        res_list.append(block_1_cross_value_states)

        block_2_cross_key_states = self.encoder.get_output_tensor(4).reshape(1, 8, 256, 64)
        block_2_cross_value_states = self.encoder.get_output_tensor(5).reshape(1, 8, 256, 64)
        res_list.append(block_2_cross_key_states)
        res_list.append(block_2_cross_value_states)

        block_3_cross_key_states = self.encoder.get_output_tensor(6).reshape(1, 8, 256, 64)
        block_3_cross_value_states = self.encoder.get_output_tensor(7).reshape(1, 8, 256, 64)
        res_list.append(block_3_cross_key_states)
        res_list.append(block_3_cross_value_states)

        block_4_cross_key_states = self.encoder.get_output_tensor(8).reshape(1, 8, 256, 64)
        block_4_cross_value_states = self.encoder.get_output_tensor(9).reshape(1, 8, 256, 64)
        res_list.append(block_4_cross_key_states)
        res_list.append(block_4_cross_value_states)

        block_5_cross_key_states = self.encoder.get_output_tensor(10).reshape(1, 8, 256, 64)
        block_5_cross_value_states = self.encoder.get_output_tensor(11).reshape(1, 8, 256, 64)
        res_list.append(block_5_cross_key_states)
        res_list.append(block_5_cross_value_states)

        return res_list

    def destory(self):
        self.encoder.destroy()
        self.decoder.destroy()
