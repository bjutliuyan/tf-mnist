import tensorflow as tf

#回归函数
def my_regression():

    #准备数据
    with tf.variable_scope("data"):
        #准备100条数据x的平均值为5.0 标准差1.0
        x = tf.random_normal([100, 1], mean = 5.0, stddev=1.0, name="x")
        #真实的关系为 y = 0.7x + 0.6
        y_true = tf.matmul(x, [[0.7]]) + 0.6

    #创建模型
    with tf.variable_scope("model"):
        #创建变量权重、偏置
        weight = tf.Variable(tf.random_normal([1,1], mean=1.0, stddev=0.1), name="weight")
        bias = tf.Variable(1.0, name="bias")
        #预测结果
        y_predict = tf.matmul(x, weight) + bias

    #计算损失
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_predict - y_true))

    #优化
    with tf.variable_scope("optimizer"):
        #梯度下降减少损失，每次学习率0.1
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    #收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weightes", weight)

    #合并变量
    merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print("初始的权重为{}, 初始的偏置为{}".format(weight.eval(), bias.eval()))
        # 添加TensorBoard记录文件
        file_write = tf.summary.FileWriter('.', graph=sess.graph)
        #循环训练线性回归模型
        for i in range(20000):
            sess.run(train_op)
            print("训练第{}次的权重为{}, 偏置为{}".format(i, weight.eval(), bias.eval()))
            # 观察每次值的变化
            # 运行merge
            summery = sess.run(merged)
            # 每次收集到的值添加到文件中
            file_write.add_summary(summery, i)

if __name__ == '__main__':
    my_regression()