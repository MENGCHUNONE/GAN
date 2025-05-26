import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # 保证使用 TF1.x 的行为


from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
import argparse
from utils import ImagePool



# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)    # 批处理大小
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--use_lsgan', type=bool, default=True)
parser.add_argument('--norm', type=str, default='instance')     # 归一化方式[instance\batch]
parser.add_argument('--lambda1', type=int, default=10)  # 循环一致性损失权重1
parser.add_argument('--lambda2', type=int, default=10)  # 循环一致性损失权重2
parser.add_argument('--learning_rate', type=float, default=2e-4)    # 学习率
parser.add_argument('--beta1', type=float, default=0.5)     # Adam优化器参数
parser.add_argument('--pool_size', type=int, default=50)    # 图像缓存池容量
parser.add_argument('--ngf', type=int, default=64)      # 生成器基础通道数

parser.add_argument('--X', type=str, default='faces/male.tfrecords')
parser.add_argument('--Y', type=str, default='faces/female.tfrecords')
parser.add_argument('--load_model', type=str, default=None)

args = parser.parse_args()


def train():
    # 模型加载模式
    if args.load_model is not None:
        checkpoints_dir = "checkpoints/" + args.load_model.lstrip("checkpoints/")
    else:
        # 新建检查点目录
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)

        # 清理冲突文件并创建目录
        if os.path.exists(checkpoints_dir) and not os.path.isdir(checkpoints_dir):
            os.remove(checkpoints_dir)

        os.makedirs(checkpoints_dir, exist_ok=True)

    graph = tf.Graph()
    with graph.as_default():
        # 初始化 CycleGAN模型
        cycle_gan = CycleGAN(
            X_train_file=args.X,
            Y_train_file=args.Y,
            batch_size=args.batch_size,
            image_size=args.image_size,
            use_lsgan=args.use_lsgan,
            norm=args.norm,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            ngf=args.ngf
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.model()  # 获取模型输出
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)     # 定义优化器

        # 汇总损失（tensorboard可视化）
        with tf.name_scope("loss"):
            tf.summary.scalar("G_loss", G_loss)
            tf.summary.scalar("D_Y_loss", D_Y_loss)
            tf.summary.scalar("F_loss", F_loss)
            tf.summary.scalar("D_X_loss", D_X_loss)

        summary_op = tf.summary.merge_all()     # 合并所有汇总操作
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)    # 日志写入器
        saver = tf.train.Saver()    # 模型保存器

    with tf.Session(graph=graph) as sess:
        # 加载或初始化模型
        if args.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])     # 恢复训练步数
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        # 启动数据队列线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # 初始化图像缓存池
            fake_Y_pool = ImagePool(args.pool_size)
            fake_X_pool = ImagePool(args.pool_size)

            while not coord.should_stop():
                # 生成假图像
                fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

                # 训练步骤，使用缓存池采样
                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
                    sess.run(
                        [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                        feed_dict={
                            cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                            cycle_gan.fake_x: fake_X_pool.query(fake_x_val)
                        }
                    )
                )
                # 记录日志
                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 100 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  G_loss   : {}'.format(G_loss_val))
                    logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                    logging.info('  F_loss   : {}'.format(F_loss_val))
                    logging.info('  D_X_loss : {}'.format(D_X_loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            logging.info('Exception: %s', str(e))
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # # 配置日志级别
    train()
