# -*- coding: utf-8 -*-
"""
@ Author: Jun Sun {Python3}
@ E-mail: sunjunee@qq.com
@ Create: 2017-08-21 15:04

Descript: Generate verify code
"""
import random, os
from captcha.image import ImageCaptcha
from multiprocessing import Pool

#10
char_set='0123456789'
numProcess = 4 

def gen_rand():
	buf = ""; max_len = random.randint(4,6)
	for i in range(max_len):
		buf += random.choice(char_set)
	return buf
				
def generateImg(imgDir, ind):
	captcha = ImageCaptcha(fonts=['fonts/Ubuntu-M.ttf']);
	theChars = gen_rand(); captcha.generate(theChars)
	img_name = '{:08d}'.format(ind) + '_' + theChars + '.png'
	img_path = imgDir + '/' + img_name
	captcha.write(theChars, img_path)

def run(num, path):
	if not os.path.exists(path):	os.mkdir(path);
	tasks = [(path, i) for i in range(num)];
	with Pool(processes = numProcess) as pool:
		pool.starmap(generateImg, tasks)		
		
if __name__=='__main__':
	run(10000, 'train')
	run(2000, 'validation')