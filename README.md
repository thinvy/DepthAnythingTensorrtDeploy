### 环境配置

1. 配置配置pytorch gpu环境，与工程下的requestment.txt

> cuda&cudnn：https://zhuanlan.zhihu.com/p/424817205 ，建议配置比显卡驱动cuda版本低一点的cuda编译器，比如我nvidia-smi查看的cuda版本是12.0，这里根据教程配置11.7。另外建议使用conda等虚拟环境，并且手动从pytorch官网下载对应gpu驱动版本的pytorch。本工程中我使用的是pytorch1.13

2. 配置tensorrt，建议配置最新的tensorrt8.6版本，对transformer block的部署优化更好

> https://zhuanlan.zhihu.com/p/392143346

1. 额外的tensorrt环境变量，设置trtexec应用的环境变量
```
# 写入 ～/.bashrc 中
export PATH=/opt/tensorrt/TensorRT-8.6.0.12/bin:$PATH
# 退出后source一下
```

1. 源码安装tensorrt python包：
https://github.com/NVIDIA/TensorRT/tree/release/8.6/tools/experimental/trt-engine-explorer

1. 配置Opencv4 C++环境

### 模型转换

1. 从pytorch模型导出onnx，这里导出vit-s编码器的depth anything
```
python3 export_onnx.py --model s
```

2. onnx 图优化
```
onnxsim \
  weights/depth_anything_vits14.onnx \
  weights/depth_anything_vits14-sim.onnx
```

3. onnx转trt engin模型文件，这里指定`--fp16`采用fp16推理精度

也可以指定`--int8 --fp16`做混合精度量化，开启后会对decoder和其他部分的conv等算子按int8量化，在pc的显卡上性能提升不明显，但在jetson这一类设备上面困难会有比较明显的提升，但不进行校准的话输出就不能看了

不能只指定`--int8`，中间vit中的一部分不能被trtexec量化到int8，会被以fp32精度推理，所以速度反而更慢了。如果想要纯int8推理，需要在pytorch导出onnx时进行ptq显式量化，并开发tensorrt相应的融合layer的插件与算子
```
trtexec \
  --onnx=weights/depth_anything_vits14-sim.onnx \
  --iterations=500 \
  --workspace=16384 \
  --percentile=99 \
  --fp16 \
  --streams=1 \
  --exportProfile=weights/depth_anything_vits14-sim-ptq-f16.profile.json \
  --exportLayerInfo=weights/depth_anything_vits14-sim-ptq-f16.graph.json \
  --saveEngine=weights/depth_anything_vits14-sim-ptq-f16.plan \
  --profilingVerbosity=detailed
```

4. 上面一步中导出了graph和prof的json文件，可以进行可视化查询模型的结构（融合算子，量化信息，prof信息等）。

先修改`trt_engin_visualize.py`中的`engine_name`，再执行
```
python3 trt_engin_visualize.py
```

**可能出现的报错**

报错1：
```
ImportError: cannot import name 'url_quote' from 'werkzeug.urls' (/home/nox/anaconda3/envs/mldev/lib/python3.8/site-packages/werkzeug/urls.py)
```
```
pip3 install werkzeug==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

报错2：visualize脚本如果出现了
```
ValueError: Could not load JSON file <_io.TextIOWrapper name='/home/nox/Workspace/nndev/depth-anything-tensorrt/weights/depth_anything_vits14-sim-ptq-f16.graph.json' mode='r' encoding='UTF-8'>
```
tensorrt8.6中生成的graph json的部分layer的metadata中有非法二进制符号，vscode中复制搜索这个符号全部删除即可



### tensorrt runtime测试
指定CMakeLists.txt中的tensorrt和cuda安装路径
指定main.cpp中模型和测速视频的路径
```
mkdir build
cd build
cmake ..
make -j32
./DepthAnythingTRTDemo
```