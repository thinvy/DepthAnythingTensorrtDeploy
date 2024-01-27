import tensorrt as trt

# ONNX文件路径和输出的TensorRT模型文件路径
onnx_model_path = 'weights/depth_anything_vits14-sim.onnx'
trt_model_path = 'weights/depth_anything_vits14-sim-fp16.trt'

# 创建一个详细日志记录的logger对象
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# 创建一个logger对象（TensorRT输出的日志信息）
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 建立TensorRT模型生成器和配置
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 解析ONNX模型
with open(onnx_model_path, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError('Failed to parse the ONNX model.')

# 创建优化配置，设置FP16模式
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

# 设置最大工作空间大小（以字节为单位）
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# 生成TensorRT模型（引擎），并序列化
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build the engine.")

# 将模型序列化为文件
with open(trt_model_path, "wb") as f:
    f.write(serialized_engine)