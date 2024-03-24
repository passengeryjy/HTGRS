import matplotlib.pyplot as plt

# #折线图
# x = [1,2,3,4]#点的横坐标
# k1 = [84.2,86.8,86.3,84.9]#线1的纵坐标
# k2 = [89.7,90.9,89.9,88.1]#线2的纵坐标
# k3 = [69.2,77.8,75.1,75.8]#线3的纵坐标
# plt.plot(x,k1,'s-',color = 'r',label="F1")#s-:方形
# plt.plot(x,k2,'o-',color = 'g',label="intra-F1")#o-:圆形
# plt.plot(x,k3,'v-',color = 'b',label="inter-F1")#o-:圆形
# plt.xlabel("Number of GCN layer")#横坐标名字
# plt.ylabel("Score")#纵坐标名字
# plt.legend(loc = "best")#图例
# plt.show()

#柱状图
#柱状图
import numpy as np
import matplotlib.pyplot as plt
# Layer1 = [84.2,89.5,74.2]
# Layer2 = [86.8,90.1,76.8]
# Layer3 = [86.3,89.9,75.1]
# Layer4 = [85.9,88.7,75.4]
# #x = ['REST','LAPT','AUTO']
# x = np.arange(3) #总共有几组，就设置成几，我们这里有三组，所以设置为3
# total_width, n = 0.8, 4    # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
# width = total_width / n
# x = x - (total_width - width) / 2
# plt.bar(x, Layer1, color = "m",width=width,label='Layer1 ')
# plt.bar(x + width, Layer2, color = "y",width=width,label='Layer2')
# plt.bar(x + 2 * width, Layer3 , color = "c",width=width,label='Layer3')
# plt.bar(x + 3 * width, Layer4 , color = "b",width=width,label='Layer4')
# plt.xlabel("(a) CDR dataset")
# plt.ylabel("Score(%)")
# plt.legend(loc = "best")
# plt.xticks([0,1,2],['F1','intra-F1','inter-F1'])
# my_y_ticks = np.arange(60.0, 95.0, 5)
# plt.ylim((60.0, 95.0))
# plt.yticks(my_y_ticks)
# plt.show()

Layer1 = [83.1,86.5,64.5]
Layer2 = [83.6,86.8,64.8]
Layer3 = [85.2,88.8,70.2]
Layer4 = [84.9,86.4,68.9]
#x = ['REST','LAPT','AUTO']
x = np.arange(3) #总共有几组，就设置成几，我们这里有三组，所以设置为3
total_width, n = 0.8, 4    # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(x, Layer1, color = "m",width=width,label='Layer1 ')
plt.bar(x + width, Layer2, color = "y",width=width,label='Layer2')
plt.bar(x + 2 * width, Layer3 , color = "c",width=width,label='Layer3')
plt.bar(x + 3 * width, Layer4 , color = "b",width=width,label='Layer4')
plt.xlabel("(b) GDA dataset")
plt.ylabel("Score(%)")
plt.legend(loc = "best")
plt.xticks([0,1,2],['F1','intra-F1','inter-F1'])
my_y_ticks = np.arange(60.0, 95.0, 5)
plt.ylim((60.0, 95.0))
plt.yticks(my_y_ticks)
plt.show()

