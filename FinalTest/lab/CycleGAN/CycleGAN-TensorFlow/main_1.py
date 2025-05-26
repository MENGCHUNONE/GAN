

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import imageio.v2 as imageio
import glob
import os




tf.compat.v1.disable_eager_execution()

image_file = 'face.jpg'
W = 256
result = np.zeros((4 * W, 5 * W, 3), dtype=np.uint8)

for gender in ['male', 'female']:
    if gender == 'male':
        images = glob.glob('../faces/male/*.jpg')
        model = '../pretrained/male2female.pb'
        r = 0
    else:
        images = glob.glob('../faces/female/*.jpg')
        model = '../pretrained/female2male.pb'
        r = 2
# 加载pb模型
    graph = tf.Graph()
    with graph.as_default():
        # 读取pb文件
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model, 'rb') as model_file:
            graph_def.ParseFromString(model_file.read())
            tf.import_graph_def(graph_def, name='')     #导入计算图

        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name('input_image:0')
            output_tensor = graph.get_tensor_by_name('output_image:0')
# 图像处理
            for i, image_path in enumerate(images[:5]):
                image = imageio.imread(image_path).astype(np.float32)
                # 执行推理
                output = sess.run(output_tensor, feed_dict={input_tensor: image})

                # 若 output 是 bytes 类型，说明是图像字节流
                if isinstance(output, bytes):
                    from PIL import Image
                    import io

                    output = Image.open(io.BytesIO(output)).convert("RGB")
                    output = np.array(output).astype(np.uint8)

                # 归一化到 0-255
                maxv = np.max(output)
                minv = np.min(output)
                output = ((output - minv) / (maxv - minv) * 255).astype(np.uint8)


                # 结果拼接
                result[r * W: (r + 1) * W, i * W: (i + 1) * W, :] = image.astype(np.uint8)
                result[(r + 1) * W: (r + 2) * W, i * W: (i + 1) * W, :] = output

if os.path.exists(image_file):
    os.remove(image_file)

imageio.imsave('CycleGAN性别转换结果2.jpg', result)

