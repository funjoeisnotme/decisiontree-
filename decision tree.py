
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus

if __name__ == '__main__':
	with open('fruit.txt', 'r',encoding='utf-8') as fr:										#加载文件
		fruit = [inst.strip().split('\t') for inst in fr.readlines()]		#处理文件
	fruit_target = []														#提取每组数据的类别，保存在列表里
	for each in fruit:
		fruit_target.append(each[-1])


	fruitLabels = ['色泽', '根蒂', '敲声', '好']			#特征标签
	fruit_list = []
	fruit_dict = {}														
	for each_label in fruitLabels:											#提取信息，生成字典
		for each in fruit:
			fruit_list.append(each[fruitLabels.index(each_label)])
		fruit_dict[each_label] = fruit_list
		fruit_list = []

	fruit_pd = pd.DataFrame(fruit_dict)									#生成pandas.DataFrame

	le = LabelEncoder()														#创建LabelEncoder()对象，用于序列化
	for col in fruit_pd.columns:											#序列化
		fruit_pd[col] = le.fit_transform(fruit_pd[col])


	clf = tree.DecisionTreeClassifier(max_depth = 4)						#创建DecisionTreeClassifier()类
	clf = clf.fit(fruit_pd.values.tolist(), fruit_target)					#使用数据，构建决策树

	dot_data = StringIO()
	tree.export_graphviz(clf, out_file = dot_data,							#绘制决策树
						feature_names = fruit_pd.keys(),
						class_names = clf.classes_,
						filled=True, rounded=True,
						special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf("tree.pdf")												#保存绘制好的决策树，以PDF的形式存储。

	print(clf.predict([[1,1,1,0]]))											#预测