# Yolov5-Oracle-recognition

本文旨在研究甲骨文智能识别中原始拓片单字自动分割与识别问题，甲骨文作为中国古代文字的重要形式，其智能化识别对于古文字学研究具有里程碑意义。
对于第一问，本文对图像的噪点进行处理。本文进行了六层的预处理建模。依次为双边滤波算法、图像灰度化、局部对比度增强算法、锐化、自适应阈值二值化方法以及灰度反转，六层预处理模型的串联应用，可以有效地提高图像的质量和清晰度。
在第二问和第三问中，我们以YOLOv5s为基础模型，并在其中融入了注意力模块，能强化对甲骨文图像中的深层特征和全局信息的捕捉,极大提高了字符检测的准确率和系统的整体效率,确保了在各种背景条件下的高性能的模型应用。
在第四问中，我们利用从前一步骤中训练的模型，从混合甲骨文数据集中框选出单个甲骨文，并应用 ResNet深度学习架构来进行文字识别。ResNet通过引入一个创新的分割注意力机制，有效地提升了网络对图像中不同区域的注意力分配，从而提高了模型对于复杂字符形态的识别能力。这一技术适用于处理甲骨文这类历史文本，因为它能够处理文本的多样性和变异性。通过这种方法，我们实现了高精度的字符识别，并将结果进行了可视化展示。
