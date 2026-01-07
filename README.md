# graduation-project-lhp
**李贺鹏的毕业设计的代码**

一、总述

​	毕业论文的名称是《基于车路协同的行人多模态轨迹预测研究》，其大体由四大模块组成，分别是目标检测与跟踪模块、坐标转换模块、轨迹融合模块、轨迹预测模块。

​	具体来说，车端与路端分别利用自身摄像头采集图像信息。首先，使用目标检测算法对视频帧进行处理，输出目标的中心点、尺寸和类别信息。随后，将视频帧输入跟踪算法经过特征提取和目标匹配后，生成跟踪结果即每个目标的历史轨迹信息。在信息采集与检测跟踪完成后，将目标在车辆与路端坐标系下的坐标转换成全局平面坐标（含时间戳），确保时间和空间的同步。接下来，将车端与路端的数据进行相似性计算，根据计算结果，将车端和路端的目标历史轨迹进行关联匹配。对于成功匹配的轨迹进行融合，使得对于每一个目标都有唯一的历史轨迹信息。最终将所有交通参与者的历史轨迹信息输入到轨迹预测模型中，在多模态轨迹输出层生成多模态的轨迹。

二、目标检测与跟踪模块

​	该模块的实现主要是在文件夹\ultralytics中，目标检测算法为YOLOv11，目标跟踪算法为BortSORT。此文件夹中的代码与YOLOv11的官方代码近乎相同，只是新增了一个\ultralytics\track.py脚本，用于提取并保存视频中的检测跟踪信息，保存目录是graduation-project-lhp\ultralytics\runs\detect\track。

​	若想试验一下模块的可用性，可运行如下指令，前提是修改好该文件中model_path和video_path的位置。

```bash
python track.py
```

三、坐标转换模块

​	由于目标检测与跟踪获取到的是二维图像2D信息，要想获得目标在三维世界的真实坐标，需要有一个坐标转换模块。而实现2D转3D的关键是深度信息，但是单目相机无法直接获取深度信息，故需要一个方法来实现深度测距。深度测距的方法使用的是MonoDepth2，在项目中的位置是\position_transform\monodepth2。

```text
├── config/ #存放标定文件
├── models/ #MonoDepth2的模型文件
├── monodepth2/ #MonoDepth2的官方代码
├── single-infrastructure-side/ #DAIR-V2X的路端设备3D感知数据
├── single-infrastructure-side-example/ #DAIR-V2X的路端设备3D感知数据例子
├── single-vehicle-side-example/ #DAIR-V2X的车端设备3D感知数据
├── test_results/ # 运行evaluate_single_image.py后的可视化结果
├── evaluate_monodepth2.py #待定
├── evaluate_single_image.py # 验证MonoDepth2可行性的文件
├── position_transform.py #待定
├── position_transform_deep.py #待定
└── position_transform_ros.py #待定
```

​	若想查看该模块的可行性，可运行如下相关指令：

```bash
python evaluate_single_image.py -h #-h指查看help,运行这个脚本需要一些参数，比如图片的路径等
```

四、轨迹融合模块

​	由于是车路协同，所以需要有一个模块对车端和路端的目标检测跟踪与坐标转换模块输出的结果进行匹配融合。通过匹配和融合多源轨迹，确保每个目标拥有唯一确定的历史轨迹信息。该模块对应的代码位置是\track_correlation。

```text
├── processed_data/ #运行tools_extract.py处理后的DAIR-V2X数据，用于后续训练轨迹预测模型
├── test_png/ # 轨迹融合的效果展示图
├── tools_extract.py #处理DAIR-V2X数据，用于训练
├── tools_match7.py # 轨迹匹配算法的主程序，用了模糊方法
├── validate_matching.py #评估匹配是否准确的脚本，V2X-Seq-TFD\cooperative-vehicle-			infrastructure
└── visualize_trajectories.py #轨迹匹配可视化脚本
```

​	若想查看该模块的可行性，需注意每个脚本中的文件路径设置正确。

五、轨迹预测模块

​	多智能体进行多模态轨迹预测的深度学习模型，借鉴于TNT，将复杂的轨迹预测任务分解为三个相互关联的阶段：目标预测阶段、轨迹生成阶段、轨迹评分阶段。模型的整体架构由四个核心模块组成：Transformer编码器、目标预测模块、轨迹生成模块、轨迹评分模块。

```text
├── checkpoints/ #保存训练后的模型和折线图等
├── config/ #训练配置
├── data/ #存放训练、推理数据
├── predictions/ # 存放推理的结果
├── processed_data/ #存放运行preprocess_data.py预处理后的pkl格式数据，方便训练与推理时快速加载
├── Transformer/ #存放预测模型的代码
├── utils/ #数据预处理工具，服务于preprocess_data.py
├── predict.py #推理脚本
├── preprocess_data.py # 数据预处理，将csv格式转换成pkl格式
├── TNT.pdf #TNT原文
└── train_tnt.py #训练文件
```
六、其他
1.若需导入本地查看，可运行
```bash
# 把代码下载到本地
git clone https://github.com/JLULHP/graduation-project-lhp.git
# 下载到本地的是main分支，需要新建自己的分支
git checkout -b 分支名字
# 这样你就拥有了自己的分支，在这个分支上进行修修改改，可用 git branch查看分支树
# 若要提交，则可运行如下指令
# 1.查看自己改了哪些代码
git status
# 2.决定哪些代码是要提交的
git add 文件夹1 文件夹2 文件1 文件2
# 3.可用git status查看当前状态，选择哪些提交之后，运行如下指令正式提交，[]可以填[add]、[fix]、[update]
git commit -m "[XXX]对你提交代码的解释"
# 4.如果你后悔了，可以询问AI如何撤回，在此之前，都可以方便地进行撤回操作
# 5.提交代码到远程仓库（github），这样你在github就可以看到你自己分支的代码
git push origin 分支名字
```
