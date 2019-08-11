import tensorflow as tf
import os


tf.app.flags.DEFINE_string("model_path", "./linear_regression/", "模型保存的路径和文件名")
FLAGS = tf.app.flags.FLAGS


def linear_regression():
    with tf.variable_scope('original_data'):
        X = tf.random_normal(shape=(100, 1), mean=2, stddev=2)
        y_true = tf.matmul(X, [[0.8]]) + 0.7

    with tf.variable_scope('linear_model'):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)))
        bias = tf.Variable(initial_value=tf.random_normal(shape=(1, 1)))
        y_predict = tf.matmul(X, weights) + bias

    with tf.variable_scope('loss'):
        error = tf.reduce_mean(tf.square(y_predict - y_true))

    with tf.variable_scope('gd_optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)

    # 合并变量
    merge = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("随机初始化的权重为%f， 偏置为%f" % (weights.eval(), bias.eval()))

        # 创建事件文件
        file_writer = tf.summary.FileWriter(logdir="./summary", graph=sess.graph)

        for i in range(300):
            sess.run(optimizer)
            print("第%d步的误差为%f，权重为%f， 偏置为%f" % (i, error.eval(), weights.eval(), bias.eval()))

            # 运行合并变量op
            summary = sess.run(merge)
            file_writer.add_summary(summary, i)
    return None


def main(argv):
    print("这是main函数")
    print(argv)
    print(FLAGS.model_path)
    linear_regression()


if __name__ == '__main__':
    tf.app.run()
    # tensorboard  --logdir="/Users/sun/PycharmProjects/TensorFlow/summary"
