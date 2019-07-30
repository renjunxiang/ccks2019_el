# ccks2019_el
CCKS 2019 Task 2: Entity Recognition and Linking

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/torch-1.1.0-brightgreen.svg)](https://pypi.org/project/torch/1.1.0)
[![](https://img.shields.io/badge/pytorch--pretrained--bert-0.6.2-brightgreen.svg)](https://pypi.org/project/pytorch-pretrained-bert/0.6.2)
[![](https://img.shields.io/badge/keras-2.2.4-brightgreen.svg)](https://pypi.org/project/keras/2.2.4)
[![](https://img.shields.io/badge/numpy-1.16.2-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.16.2)

## **项目简介**
CCKS 2019 中文短文本的实体链指 (CCKS 2019 Task 2: Entity Recognition and Linking)，<https://biendata.com/competition/ccks_2019_el/>，第十名代码分享。<br>

## **整体思路**
* 实体识别采用BERT+RNN+指针网络<br>
* 实体消岐由于BERT太慢，采用预训练词向量+CNN+二分类器<br>

## **运行方式**
### **获取预训练数据**
1.BERT、BERT-wwm：<https://github.com/ymcui/Chinese-BERT-wwm><br><br>
2.ERBIE：<https://github.com/PaddlePaddle/LARK>，考虑到版权及作者申明，pytorch版的权重请自行查找<br><br>
3.词向量：<https://github.com/Embedding/Chinese-Word-Vectors><br><br>
在此表示感谢！<br>

### **数据预处理**
1.运行脚本./preprocess.py在./data_deal中生成预处理结果，为了提高训练和推断速度，预处理的时候会保留文本、词典编码、实体位置和链接的kb_id，只尝试了保留频数最多的10000个字，提供部分预处理数据（tokenizer.pkl和develop_data.pkl供参考，建议自己重新跑一下），主要包含：<br><br>
* tokenizer.pkl，文本编码完整信息<br>
* word_index.pkl，文本编码词典信息，预处理dev、eval数据会需要<br>
* alias_data.pkl，知识库信息，会生成“实体-id-描述”的字典，用于训练过程的消岐抽样和推断过程的实体过滤<br>
* subject_data.pkl，知识库信息，会生成“id-描述”的字典，用于训练过程的消岐抽样<br>
* ./10000/xxx_data.pkl，预处理转换，编码保留前10000个字，后续的训练和推断都基于此<br><br>

### **获取嵌入层权重**
1.下载词向量，运行脚本./embedding.py，抽取出字向量，保存在./data_deal中，以便后续nn.embedding的时候可以导入<br><br>

### **训练模型**
这里将网络拆开，实际训练中由于实体识别和实体消岐无法同时达到最佳，所以需要先训练实体识别，再载入实体识别最优模型权重训练消岐<br><br>
1.将bert预训练放入pretrain中，也可以自行修改训练脚本中的位置<br><br>
2.运行脚本./train_part_ner.py，输入网络参数，实体识别模型保存在./results_ner中：python train_part_ner.py --cuda 0 --pretrain bert --num_layers 3 --hidden_dim 768 --loss_weight 2 --epochs 3 --k 0.820<br><br>
3.运行脚本./train_part_link.py，输入实体识别的模型id，完整模型保存在./results中：python train_part_link.py --cuda 0 --pretrain bert --num_layers 3 --hidden_dim 768 --loss_weight 2 --ner_id 2 --num_words 10000 --max_len 400 --epochs 25 --lr 0.001 --k 0.9030 --n 2<br><br>
4.模型和每次的预测结果都保存在./results/10000中，后续ensemble需要<br><br>

### **测试模型**
1.运行脚本./predict.py，推断eval数据集<br><br>
2.运行脚本./submit_test.py，对eval结果做ensemble用于提交<br><br>

## **其他说明**
1.为了减小模型大小，bert预训练是在dateset中计算好语义再输入模型，而不是模型中嵌入；<br><br>
2.消岐过程发现bert和词向量+CNN好像没太大区别；<br><br>
3.类似书名号等规则并没有太大的提升（submit.py就是最后投票融合的，也没什么有效的规则和方法），感觉标注本身就不太统一<br><br>
PS:比赛代码在服务器上执行，本代码如有问题请先debug，再次对主办方开放数据集表示感谢！<br><br>

## **最终成绩**
![](https://github.com/renjunxiang/ccks2019_el/blob/master/picture/score.png)
