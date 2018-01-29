import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from include.data import getDataSet
from include.model import model
import tkinter

CIFAR=100

test_x, test_y, test_l = getDataSet("test" + str(CIFAR), CIFAR)
x, y, output, global_step, y_pred_cls = model(CIFAR)

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = CIFAR
_SAVE_PATH = "./tensorboard/cifar-" + str(CIFAR) + "/"

saver = tf.train.Saver()
sess = tf.Session()

try:
    print("Attempt to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Success! Restoring checkpoint from:", last_chk_path)
except:
    print("Failed! Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

i = 0
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
while i < len(test_x):
    j = min(i + _BATCH_SIZE, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]
    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
    i = j

correct = (np.argmax(test_y, axis=1) == predicted_class)
acc = correct.mean()*100
correct_numbers = correct.sum()
cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)

root = tkinter.Tk()
for i in range(_CLASS_SIZE):
    for j in range(_CLASS_SIZE):
        b = tkinter.Entry(root, text="", width = 10)
        b.insert(0, cm[i][j])
        b.configure(state='readonly')
        b.grid(row=i, column=j)

for i in range(_CLASS_SIZE):
    b = tkinter.Entry(root, text="", width=10)
    b1 = tkinter.Entry(root, text="", width=10)
    b.insert(0, test_l[i])
    b1.insert(0, test_l[i])
    b.configure(state='readonly')
    b1.configure(state='readonly')
    b.grid(row=i, column=_CLASS_SIZE)
    b1.grid(row=_CLASS_SIZE, column=i)

accuracy = tkinter.Entry(root, text="", width = 10)
accuracy.insert(0, "Accuracy:" )
accuracy.configure(state='readonly')
accuracy.grid(row=_CLASS_SIZE + 1, column=0)

accuracy = tkinter.Entry(root, text="", width = 10)
accuracy.insert(0, acc)
accuracy.insert(tkinter.END, '%')
accuracy.configure(state='readonly')
accuracy.grid(row=_CLASS_SIZE + 1, column=1)

accuracy = tkinter.Entry(root, text="", width = 10)
accuracy.insert(0, "Correct:" )
accuracy.configure(state='readonly')
accuracy.grid(row=_CLASS_SIZE + 2, column=0)

accuracy = tkinter.Entry(root, text="", width = 10)
accuracy.insert(0, correct_numbers)
accuracy.insert(tkinter.END, '/')
accuracy.insert(tkinter.END, len(test_x))
accuracy.configure(state='readonly')
accuracy.grid(row=_CLASS_SIZE + 2, column=1)

tkinter.mainloop()
sess.close()
