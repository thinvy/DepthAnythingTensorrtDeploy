[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-76B900)](https://developer.nvidia.com/tensorrt)
[![GitHub Repo stars](https://img.shields.io/github/stars/thinvy/DepthAnythingTensorrtDeploy)](https://github.com/fabio-sim/Depth-Anything-ONNX/stargazers)
[![GitHub all releases](https://img.shields.io/github/downloads/thinvy/DepthAnythingTensorrtDeploy/total)](https://github.com/fabio-sim/Depth-Anything-ONNX/releases)

# Depth Anything Tensorrt Deploy

NVIDIA TensorRT compatible implementation of [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://github.com/LiheYoung/Depth-Anything).

<p align="center"><img src="[assets/sample.png](https://github.com/LiheYoung/Depth-Anything/blob/main/assets/teaser.png?raw=true)" width=90%>


### 环境配置

1. 配置配置pytorch gpu环境，与工程下的requestment.txt

> cuda&cudnn：https://zhuanlan.zhihu.com/p/424817205 ，建议配置比显卡驱动cuda版本低一点的cuda编译器，比如我nvidia-smi查看的cuda版本是12.0，这里根据教程配置11.7。另外建议使用conda等虚拟环境，并且手动从pytorch官网下载对应gpu驱动版本的pytorch。本工程中我使用的是pytorch1.13

2. 配置tensorrt，建议配置最新的tensorrt8.6版本，对transformer block的部署优化更好

> https://zhuanlan.zhihu.com/p/392143346

3. 额外的tensorrt环境变量，设置trtexec应用的环境变量
```
# 写入 ～/.bashrc 中
export PATH=/opt/tensorrt/TensorRT-8.6.0.12/bin:$PATH
# 退出后source一下
```

4. 源码安装tensorrt python包：
```
https://github.com/NVIDIA/TensorRT/tree/release/8.6/tools/experimental/trt-engine-explorer
```

5. (c++ runtime 测试依赖) 配置Opencv4 C++环境

6. (可选)下载[转换好的vit-s的预训练模型](https://drive.google.com/drive/folders/1qPGPQcSSnHHeMq0eU7Vrm3DD_9dTAD-7?usp=sharing)放在weigth文件夹中，用于直接测试（带有int8量化的模型由于没有校准，输出不可用，建议pc端测试fp16部署，jetson平台测试int8-fp16混合精度量化）


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
**性能参考**
测试环境：PC (14700k + RTX3080TI); Ubuntu20.04 cuda11.7 tensorrt8.6
性能参考：
| weight | quantize  | time  |
| vit-s  | fp16      | 2.95ms|
| vit-s  | int8+fp16 | 2.77ms|


### Acknowledgement
- Depth-Anything : https://github.com/LiheYoung/Depth-Anything
- Depth Anything ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- Depth Anything TensorRT: https://github.com/spacewalk01/depth-anything-tensorrt


### Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [Depth Anything](https://arxiv.org/abs/2401.10891) and [DINOv2](https://arxiv.org/abs/2304.07193). Lastly, if the ONNX versions helped you in any way, please also consider starring this repository.

```bibtex
@article{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      journal={arXiv:2401.10891},
      year={2024}
}
```

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
