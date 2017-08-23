# -*- coding: utf-8 -*-
"""
@ Author: Jun Sun {Python3}
@ E-mail: sunjunee@qq.com
@ Create: 2017-08-21 20:18

Descript: Training LSTM with CTC loss For verify codes
"""

import os
import numpy as np
import tensorflow as tf
import time
import utils

restore = True;
checkpoint_dir = './checkpoint/';
initial_learning_rate = 1e-3;

num_layers = 2;
num_hidden = 48;
num_epochs = 100;
batch_size = 50;
save_steps = 200;
validation_steps = 200;

decay_rate = 0.9;
decay_steps = 1000;

beta1 = 0.9;
beta2 = 0.999;
momentum = 0.9;

log_dir = './log';

image_width = 120;
image_height = 45;

num_features = image_height;
num_classes = 10 + 1 + 1;

class Graph(object):
	def __init__(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			# LSTM网络的输入，[batch_size, max_timesteps, num_features]
			# num_features可理解为图片的一列，max_timesteps可理解为图片的列数
			self.inputs = tf.placeholder(tf.float32, [None, None, num_features]);
			shape = tf.shape(self.inputs);	batch_size, _ = shape[0], shape[1];
			
			# 产生一个ctc_loss需要的sparse_placeholder
			self.labels = tf.sparse_placeholder(tf.int32)
			
			# 一个batch中每个样本序列的长度，是一维向量
			self.seq_len = tf.placeholder(tf.int32, [None])
			
			# 多层RNN结构（两层LSTM堆叠），隐藏状态为num_hidden，dynamic_rnn使得rnn的输入序列可以变长
			stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)
			outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)
			outputs = tf.reshape(outputs, [-1, num_hidden])

			# 连接一个全连接层
			W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1, dtype=tf.float32), name='W')
			b = tf.Variable(tf.constant(0., dtype = tf.float32, shape=[num_classes], name='b'))
			logits = tf.matmul(outputs, W) + b

			# 将结果变成[batch_size, -1, num_classes]的形状
			logits = tf.reshape(logits, [batch_size, -1, num_classes])

			# 将时间放到第一维
			logits = tf.transpose(logits, (1, 0, 2))
			
			# 定义CTC loss
			self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
			self.cost = tf.reduce_mean(self.loss)
			
			self.global_step = tf.Variable(0, trainable=False)
			self.learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step, decay_steps, decay_rate, staircase=True)
#			self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum = momentum,use_nesterov=True).minimize(self.cost,global_step=self.global_step)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,beta1=beta1,beta2=beta2).minimize(self.cost,global_step=self.global_step)
			#分类结果
			self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
			self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
			
			#分类错误率
			self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
			
			tf.summary.scalar('cost', self.cost)
			self.merged_summay = tf.summary.merge_all()

def train(train_dir=None, val_dir=None):
	#载入训练、测试数据
	print('Loading training data...')
	train_feeder = utils.DataIterator(data_dir=train_dir)
	print('Get images: ', train_feeder.size)

	print('Loading validate data...')
	val_feeder=utils.DataIterator(data_dir=val_dir)
	print('Get images: ', val_feeder.size)

	#定义网络结构
	g = Graph()
	
	#训练样本总数
	num_train_samples = train_feeder.size
	#每一轮(epoch)样本可以跑多少个batch
	num_batches_per_epoch = int(num_train_samples / batch_size)
	
	with tf.Session(graph = g.graph) as sess:
		sess.run(tf.global_variables_initializer())
		
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
		
		# restore = True 加载模型
		if restore:
			ckpt = tf.train.latest_checkpoint(checkpoint_dir)
			if ckpt:
				# global_step也会被加载
				saver.restore(sess, ckpt);
				print('restore from the checkpoint{0}'.format(ckpt))

		print('============begin training============')
		# 获取一个batch的验证数据，制作成placeholder的输入格式
		val_inputs, val_seq_len, val_labels = val_feeder.input_index_generate_batch()
		val_feed = {g.inputs: val_inputs, g.labels: val_labels, g.seq_len: val_seq_len}
		
		start_time = time.time();
		for cur_epoch in range(num_epochs):	#按照epoch进行循环
			shuffle_idx = np.random.permutation(num_train_samples)	#将训练样本的index打乱
			train_cost = 0;

			for cur_batch in range(num_batches_per_epoch):	#对于当前epoch中的每个bacth进行训练
				# 获取一个batch的训练样本，制作成placeholder的输入格式
				indexs = [shuffle_idx[i % num_train_samples] for i in range(cur_batch * batch_size, (cur_batch+1) * batch_size)];
				batch_inputs, batch_seq_len, batch_labels = train_feeder.input_index_generate_batch(indexs);
				feed = {g.inputs: batch_inputs, g.labels:batch_labels, g.seq_len:batch_seq_len};

				# 训练run
				summary_str, batch_cost, step, _ = sess.run([g.merged_summay, g.cost, g.global_step, g.optimizer], feed)
				# 计算损失
				train_cost += batch_cost;

				# 打印
				if step % 50 == 1:
					end_time = time.time();
					print('No. %5d batches, loss: %5.2f, time: %3.1fs' % (step, batch_cost, end_time-start_time));
					start_time = time.time();
				
				#验证集验证、保存checkpoint：
				if step % validation_steps == 1:
					if not os.path.isdir(checkpoint_dir):	os.mkdir(checkpoint_dir);
					saver.save(sess,os.path.join(checkpoint_dir, 'ocr-model'), global_step=step)
					
					#解码的结果：
					dense_decoded, lastbatch_err, lr = sess.run([g.dense_decoded, g.lerr, g.learning_rate], val_feed)
					acc = utils.accuracy_calculation(val_feeder.labels, dense_decoded, ignore_value=-1, isPrint=False)
					print('-After %5d steps, Val accu: %4.2f%%' % (step, acc));

if __name__ == '__main__':
	train(train_dir='train', val_dir='validation')
