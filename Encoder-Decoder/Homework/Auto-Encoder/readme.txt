源码说明：
1、auto_encoder.py可直接运行，若想自己训练一次可取消第23行new_models()的注释。
2、mnist_classification.py是一个使用原图像784维进行分类的demo，可直接运行。

作业：
1、大家可以试着将MINST分类模型中使用的784维换成auto_encoder中生成的code，看看效果如何。注意auto_encoder默认降成1维，和原图784维的分类效果谁好我也不知道。训练中请尝试改变code的维度。
2、有时间的话可以试试用denoise auto-encoder来降维，看看效果。