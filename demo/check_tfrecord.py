import tensorflow as tf

for example in tf.python_io.tf_record_iterator('../res/IF-Mel/fold2-00000-of-00002'):
    print(tf.train.Example.FromString(example))