import os
#import config
import numpy as np
import cv2

#10 digit + blank + space
num_classes = 10 + 1 + 1
channel = 1
image_width = 120
image_height = 45
num_features = image_height * channel
SPACE_INDEX = 0
SPACE_TOKEN = ''
maxPrintLen = 10

charset = '0123456789'
encode_maps = {};	decode_maps = {};
for i, char in enumerate(charset, 1):
    encode_maps[char] = i;	decode_maps[i] = char;
encode_maps[SPACE_TOKEN]=SPACE_INDEX
decode_maps[SPACE_INDEX]=SPACE_TOKEN

#数据读取、按index取batch
class DataIterator:
	def __init__(self, data_dir):
		self.image_names = []; self.image = [];	self.labels=[];
		file_list = os.listdir(data_dir);
		for file_path in file_list:
			if file_path[-3:] != 'png':	#只要png的图片
				continue;
			image_name = os.path.join(data_dir, file_path)
			self.image_names.append(image_name)

			im = cv2.imread(image_name, 0).astype(np.float32) / 255.
			im = cv2.resize(im,(image_width, image_height))
			im = im.swapaxes(0,1)	#变成120*45并转成45*120
			self.image.append(np.array(im))

			# 验证码内容
			code = file_path.split('_')[1].split('.')[0]
			code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
			self.labels.append(code)

	@property
	def size(self):
		return len(self.labels)

	def the_label(self,indexs):
		labels=[]
		for i in indexs:
			labels.append(self.labels[i])
			return labels
	
	#输入index，返回图片数据
	def input_index_generate_batch(self, index=None):
		if index:
			image_batch = [self.image[i] for i in index];
			label_batch = [self.labels[i] for i in index];
		else:
			image_batch = self.image
			label_batch = self.labels

		def get_input_lens(sequences):
			lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
			return sequences, lengths
			
		batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
		batch_labels = sparse_tuple_from_label(label_batch)
		return batch_inputs, batch_seq_len, batch_labels

def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint = True):
	count = 0
	for i,origin_label in enumerate(original_seq):
		decoded_label  = [j for j in decoded_seq[i] if j != ignore_value]
		if origin_label == decoded_label: count+=1;
	return count * 1.0 / len(original_seq)

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = [];	values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64);values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def pad_input_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

