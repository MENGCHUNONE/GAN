import tensorflow.compat.v1 as tf
import random
import os

tf.disable_eager_execution()  # 必须加上，保持 TF1.x 行为

try:
    from os import scandir
except ImportError:
    from scandir import scandir

# 替代 tf.flags 的写法（改为 argparse）
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--X_input_dir', type=str, default='faces/male')
parser.add_argument('--Y_input_dir', type=str, default='faces/female')
parser.add_argument('--X_output_file', type=str, default='faces/male.tfrecords')
parser.add_argument('--Y_output_file', type=str, default='faces/female.tfrecords')
args = parser.parse_args()


def data_reader(input_dir, shuffle=True):
    file_paths = []
    input_dir = os.path.abspath(input_dir)
    print("读取文件夹路径：", input_dir)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"目录不存在: {input_dir}")

    # 扫描目录获取图像文件
    for img_file in scandir(input_dir):
        if img_file.name.lower().endswith(('.jpg', '.jpeg', '.png')) and img_file.is_file():
            file_paths.append(img_file.path)

    # 随机打乱
    if shuffle:
        random.seed(12345)  # 固定随机种子保证可复现
        random.shuffle(file_paths)

    print(f"找到 {len(file_paths)} 个图像文件")
    return file_paths



def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
    file_name = os.path.basename(file_path)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/file_name': _bytes_feature(tf.compat.as_bytes(file_name)),
        'image/encoded_image': _bytes_feature(image_buffer)
    }))
    return example


def data_writer(input_dir, output_file):
    file_paths = data_reader(input_dir)
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = tf.io.TFRecordWriter(output_file)
    for i, file_path in enumerate(file_paths):
        with tf.io.gfile.GFile(file_path, 'rb') as f:
            image_data = f.read()

        if not image_data:
            print(f" 跳过空文件: {file_path}")
            continue
        else:
            print(f" 读取成功: {file_path}, 大小: {len(image_data)} 字节")

        try:
            example = _convert_to_example(file_path, image_data)
            writer.write(example.SerializeToString())
            print(f"已写入 TFRecord: {file_path}")
        except Exception as e:
            print(f" 写入失败: {file_path}, 错误: {e}")
            continue

        if i % 500 == 0:
            print("Processed {}/{}.".format(i + 1, len(file_paths)))
    writer.close()
    print("Done.")



def main():
    print("Convert X data to tfrecords...")
    data_writer(args.X_input_dir, args.X_output_file)
    print("Convert Y data to tfrecords...")
    data_writer(args.Y_input_dir, args.Y_output_file)


if __name__ == '__main__':
    main()
