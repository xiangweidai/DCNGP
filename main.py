# -*-coding:utf-8-*-
from typing import Dict, Text
from scipy import ndimage
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import random
import attr
import evaluation_util
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
from joblib import dump
from absl import app
from absl import flags
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from tcn import TCN
import matplotlib.pyplot as plt
import seaborn as sns
# flag函数用于在命令行中我们为了执行train.py文件，在命令行中输入 python train.py --data_dir ./data/SoyNAM --result_dir ./results --dataset_type height进行py文件的运行
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data', 'Dirctory of dataset')  # 定义数据集地址 默认为./data/SoyNAM
flags.DEFINE_string('result_dir', './result', 'Dirctory of output')  # 定义结果保存地址 默认为./results
flags.DEFINE_string('dataset_type', 'twg',
                    'Type of dataset .')  # 定义数据集类型 默认为height
flags.DEFINE_integer('batch_size', 10, 'Batch size for training.')  # 默认值为10
flags.DEFINE_integer('early_stopping_patience', 16, 'Patience for early stopping.')  # 默认值为16
flags.DEFINE_integer('reduce_lr_patience', 10, 'Patience for reducing learning rate.')  # 默认值为10
flags.DEFINE_float('leaky_alpha', 0.1, 'Alpha value for LeakyReLU.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')  # 添加新的FLAG来定义学习率
flags.DEFINE_string('scaler_path', './result/scaler.joblib', 'Path to save the StandardScaler state')
@attr.s(auto_attribs=True)
class experimental_parameters:  # 实验参数
    # 训练超参数设置
    learning_rate: float = 0.0001
    epochs: int = 250
    optimizer: Text = 'adam'
    loss: Text = 'mse'
    metrics: Text = 'mae'  # 同时计算平均绝对误差和均方根误差作为评估指标
    batch_size: int= 10
    @classmethod
    def from_dict(cls, experimental_parameters: Dict):
        return cls(
            learning_rate=experimental_parameters['learning_rate'],
            epochs=experimental_parameters['epochs'],
            optimizer=experimental_parameters['optimizer'],
            loss=experimental_parameters['loss'],
            metrics=experimental_parameters['metrics'],
            batch_size=experimental_parameters['batch_size'],
        )
def get_experimental_parameters_from_flags():
    return experimental_parameters(
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate
    )
def DCNGP_model(input_data: np.array) -> keras.Model:
    num_classes = 1
    inputs = keras.Input(shape=(input_data.shape[1], num_classes), name="input")
    x = tf.keras.layers.GaussianNoise(stddev=0.0001)(inputs)
    x = keras.layers.Conv1D(filters=8, kernel_size=4,
                            padding='same',
                            bias_regularizer=keras.regularizers.l2(0.01),
                            kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU(alpha=FLAGS.leaky_alpha)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(filters=16, kernel_size=4,
                            padding='same',
                            bias_regularizer=keras.regularizers.l2(0.01),
                            kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU(alpha=FLAGS.leaky_alpha)(x)
    x = keras.layers.Conv1D(filters=16, kernel_size=4,
                            padding='same',
                            bias_regularizer=keras.regularizers.l2(0.01),
                            kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.LeakyReLU(alpha=FLAGS.leaky_alpha)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    outputs = keras.layers.Dense(1, activation='linear', name='output',
                                 kernel_regularizer=keras.regularizers.l2(0.01))(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
def correlationMetric(x, y, axis=-2):
    x = tf.convert_to_tensor(x)
    y = tf.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    xvar = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    return  corr
def load_data(filename):
    data = pd.read_csv(filename)
    features = data.drop("label", axis=1)
    labels = data["label"]
    x = features.values
    y = labels.values
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    #label_scaler = StandardScaler()
    #y = label_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    return x, y
from sklearn.model_selection import KFold
sns.set(style="whitegrid")
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", font="Times New Roman")  # 设置seaborn风格和字体


def format_equation(slope, intercept):
    sign = '+' if intercept >= 0 else '-'
    return f'y = {slope:.2f}x {sign} {abs(intercept):.2f}'
#画核密度图
'''
def create_density_plot(y_true, y_pred, output_filename):
    plt.figure(figsize=(8, 8))
    plt.rcParams["font.family"] = "Times New Roman"
    sns.kdeplot(y_true, y_pred, cmap="Blues", shade=True, clip=[[-2.5, 2.5], [-2.5, 2.5]])  # 使用clip参数来裁剪数据

    x_min, x_max = -2, 2
    y_min, y_max =-2,2
    sns.lineplot([x_min, x_max], [y_min, y_max], color="green")  # 调整后的y = x 线

    # 计算并绘制回归线（红色）
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    y_fit_start = slope * (-1.5) + intercept
    y_fit_end = slope * 1 + intercept
    sns.lineplot([-2.5, 2.5], [y_fit_start, y_fit_end], color="red")

    equation_str = format_equation(slope, intercept)
    plt.annotate(equation_str, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, color="black", backgroundcolor='white', fontfamily="Times New Roman")
    plt.annotate('PCC=0.61', xy=(0.05, 0.90), xycoords='axes fraction',
                 fontsize=12, color="black", backgroundcolor='white', fontfamily="Times New Roman")

    plt.xlabel('True', fontfamily="Times New Roman")
    plt.ylabel('Pred', fontfamily="Times New Roman")
    plt.title('True vs. Pred', fontfamily="Times New Roman")
    plt.xlim([-2.5, 2.5])  # 限制x轴的范围
    plt.ylim([-1, 1])  # 限制y轴的范围
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()
'''
def train_model(batch_size, early_stopping_patience, reduce_lr_patience) -> keras.Model:
    training_filename = os.path.join(FLAGS.data_dir, "%s.train.csv" % (FLAGS.dataset_type))
    model_prefix = FLAGS.dataset_type # 提取前缀
    test_filename = os.path.join(FLAGS.data_dir, "%s.test.csv" % (FLAGS.dataset_type))
    x_all, y_all = load_data(training_filename)  # Load training data
    x_test, y_test = load_data(test_filename)  # Load test data
    experimental_params = get_experimental_parameters_from_flags()

    models = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

    label_scaler = StandardScaler()  # 初始化标准化器
    label_scaler.fit(y_all.reshape(-1, 1))
    scaler_path = FLAGS.scaler_path
    dump(label_scaler, scaler_path)
    for train_index, valid_index in kfold.split(x_all):
        x_train, x_valid = x_all[train_index], x_all[valid_index]
        y_train, y_valid = y_all[train_index], y_all[valid_index]
        model = DCNGP_model(x_all)
        model.compile(loss=experimental_params.loss,
                      optimizer=experimental_params.optimizer,
                      metrics=[experimental_params.metrics])

        early_stopping = EarlyStopping(monitor='val_mae',
                                       patience=FLAGS.early_stopping_patience,
                                       restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.1, patience=FLAGS.reduce_lr_patience, verbose=1,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=1e-6)
        callbacks = [early_stopping, reduce_lr]
        model.summary()
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=experimental_params.epochs,
                  validation_data=(x_valid, y_valid),
                  verbose=1,
                  callbacks=callbacks)
        models.append(model)
    predictions = []
    for model in models:
        pred = model.predict(x_test)
        predictions.append(pred)

    output_filename = os.path.join(FLAGS.result_dir, f"pcc_{FLAGS.dataset_type}.csv")
    pred_test_y = np.mean(predictions, axis=0)
    pred_test_y = label_scaler.inverse_transform(pred_test_y)
    evaluation_util.save_pcc(pred_test_y, y_test, output_filename)
    #create_density_plot(y_test, pred_test_y.flatten(), output_density_filename)  画核密度图
    final_model = DCNGP_model(x_all)
    final_model.compile(loss=experimental_params.loss, optimizer=experimental_params.optimizer,
                        metrics=[experimental_params.metrics])
    final_model.fit(x_all, y_all, batch_size=batch_size, epochs=experimental_params.epochs, verbose=1)
    # 保存最终模型
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    final_model_path = os.path.join(FLAGS.result_dir, f'{model_prefix}_model.h5')
    final_model.save(final_model_path)
    return final_model
def main(argv):
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)
    experimental_params = get_experimental_parameters_from_flags()
    train_model(experimental_params.batch_size, FLAGS.early_stopping_patience, FLAGS.reduce_lr_patience)
if __name__ == "__main__":
    app.run(main)



