import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 用于设置将记录哪些消息的阈值
old_v = tf.logging.get_verbosity()
# 设置日志反馈模式
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 定义每次训练批次为100
batch_size = 100
# 计算共训练多少批次
n_batch = mnist.train.num_examples // batch_size


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置值初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积函数，其中x是输入，W是权重，
# strides表示步长，或者说是滑动速率，包含长宽方向
# 的步长。padding表示补齐数据。 目前有两种补齐方式，
# 一种是SAME，表示补齐操作后（在原始图像周围补充0），实
# 际卷积中，参与计算的原始图像数据都会参与。一种是VALID，
# 补齐操作后，进行卷积过程中，原始图片中右边或者底部
# 的像素数据可能出现丢弃的情况。
def conv2d(x, W):
    # 步长为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 这步定义函数进行池化操作，在卷积运算中，是一种数据下采样的操作，
# 降低数据量，聚类数据的有效手段。常见的池化操作包含最大值池化和均值池化。
# 这里的2*2池化，就是每4个值中取一个，池化操作的数据区域边缘不重叠。
# 函数原型：def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)。
# 默认NHWC，表示4维数据，[batch,height,width,channels].
# 下面函数中的ksize，strides中，每次处理都是一张图片，对应的处理数据是一个通道
# （例如，只是黑白图片）。长宽都是2，表明是2*2的 池化区域，也反应出下采样的速度。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 将输入tensor进行形状调整，调整成为一个28*28的图片，
# 因为输入的时候x是一个[None,784]，有与reshape的输入项shape
# 是[-1,28,28,1]，后续三个维度数据28,28,1相乘后得到784，
# 所以，-1值在reshape函数中的特殊含义就可以映射程None。即输入图片的数量batch。
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一层卷积的权重和偏执值
# 5*5的采样窗口（卷积核），1个输入通道，输出32个通道（第二个卷积层有32个卷积核）
W_conv1 = weight_variable([5, 5, 1, 32])
# 偏置量定义，偏置的维度是32
b_conv1 = bias_variable([32])
# 将2维卷积的值加上一个偏置后的tensor，进行relu激活函数操作
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二层卷积的权重和偏执值
# 5*5的采样窗口，32个输入通道，64个输出通道
W_conv2 = weight_variable([5, 5, 32, 64])
# 偏置量定义，偏置的维度是64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 初始化第一个全连接层的权值，
# 图片尺寸减小到7x7，加入一个有1024个神经元的全连接层，用于处理整个图片。
# 把池化层输出的张量reshape成一些向量，乘上权重矩阵，
# 加上偏置，然后对其使用ReLU激活操作。
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# 将第二层池化后的数据进行变形
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 进行矩阵乘，加偏置后进行relu激活
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 对第二层卷积经过relu后的结果，基于tensor值keep_prob进行保留
# 这个是为了防止过拟合，快速收敛。
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 计算输出
# 最后，添加一个softmax层，就像前面的单层softmax regression一样。
# softmax是一个多选择分类函数，其作用和sigmoid这个二值
# 分类作用地位一样，在我们这个例子里面，softmax输出是10个（对应10个数字）。
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                                          logits=prediction))
# 使用AdamOptimizer进行优化
# 此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 记录预测值和标签值对比结果
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(12):
        # 程序循环训练12次，
        for batch in range(n_batch):
            # 程序循环一次训练n_bath批次数据
            # 每批次100个图片数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 此步主要是用来训练W和bias用的。基于似然估计函数进行梯度下降，
            # 收敛后，就等于W和bias都训练好了，keep_prob=0.7表示70%的数据参与
            # 计算，防止过拟合和减少计算量
            sess.run(train_step, feed_dict={x: batch_xs,
                                            y: batch_ys,
                                            keep_prob: 0.7})
        # 用训练好的模型（权重W，偏执值b）对测试图片和测试标签值以及
        # 给定的keep_prob进行feed操作，进行计算测试识别率。keep_prob=1.0
        # 表示所有的数据都参与运算
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y: mnist.test.labels,
                                            keep_prob: 1.0})
        print("Iter "+str(epoch)+", Testing Accuracy= "+str(acc))