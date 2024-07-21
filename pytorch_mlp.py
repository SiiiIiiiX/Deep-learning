import copy
from matplotlib import pyplot as plt
from matplotlib import animation

#训练数据集
training_set = [[(1,2),1],[(2,3),1],[(3,1),-1],[(4,2),-1]]
#参数初始化
w=[0,0]
b=0
#用来记录每次更新过后的w,b
history=[]

def update(item):
    '''
     随机梯度下降更新参数
     ：param item :参数是分类错误的点
     ：return：无返回值
    '''
    #把w,b,history声明为全局变量
    global w,b,history
    #根据误分类点更新参数，这里学习效率为1
    w[0] += 1 * item[1] * item[0][0]
    w[1] += 1 * item[1] * item[0][1]
    #将每次更新过后的w,b记录在history数组中
    history.append([copy.copy(w), b])

def cal(item):
    '''
    计算item到超平面的距离，输出yi(w*xi+b)
    根据此结果判断分类是否错误，如果yi(w*xi+b)>0，则分类错误
    '''
    res = 0
    #迭代item每个做表，对于本文数据则有两个坐标x1和x2
    for i in range(len(item[0])):
        res += item[0][i] * w[i]
    res += b
    #乘以公式中的yi
    res *=item[1]
    return res

def check():
    '''
        超平面是否已将样本正确分类
        :return：如果已经正确分类则返回True
    '''
    flag = False
    for item in training_set:
        #如果分类错误
        if cal(item) <= 0:
            #将flag设置为Ture
            flag = True
            #用错误分类点更新参数
            update(item)
    #如果没有分类错误的点
    if not flag:
        #输出达到正确结果时参数的值
        print("最终结果:w:" + str(w) + "b:" + str(b))
    #如果返回Ture:分类正确，False:分类错误
    return flag

if __name__ == "__main__":
    #迭代1000遍
    for i in range(1000):
        #如果已分类正确，则结束迭代
        if not check():
            break
    #以下过程将迭代过程可视化
    #首先建立想要做成动画的图像figure，坐标轴和plot element
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)
    label = ax.text([], [], '')


    def init():
        #在坐标轴中把4个数据点画出来
        line.set_data([],[])
        x, y, x_, y_ = [],[],[],[]
        for p in training_set:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])

        plt.plot(x,y,'bo',x_,y_,'rx')
        plt.axis([-6,6,-6,6])
        plt.grid(True)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Perception Algorithm ')
        return line, label

    def animate(i):
        global history,ax,line,label
        w=history[i][0]
        b=history[i][1]
        if w[1] == 0:
            return line, label

        x1 = -7
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7
        y2 = -(b + w[0] * x2) / w[1]

        line.set_data([x1,x2],[y1,y2])

        x1=0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(history[i])
        label.set_position([x1,y1])
        return line, label

    print("参数w,b更新过程：",history)
    anim =animation.FuncAnimation(fig,animate,init_func=init,frames=len(history),interval=1000,repeat=True,blit=True)
    plt.show()





