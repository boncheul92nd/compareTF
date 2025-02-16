import tensorflow as tf

# Function to tell TensorFlow how to read a single image from TFRecord file(s)
def getImage(filenames, nepochs=None, mtlclasses=0):
    # convert filenames to a queue for an input pipeline.
    print('getImage ' + str(filenames))
    filenameQ = tf.train.string_input_producer(filenames,num_epochs=nepochs)
 
    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example 
    key, fullExample = recordReader.read(filenameQ)
    print("TFRecordReader produced " + str(recordReader.num_records_produced()) + " records")

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/channels':  tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/class/label_2': tf.FixedLenFeature([],tf.int64, default_value=0),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded_ch1': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded_ch2': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })


    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer_channel1 = features['image/encoded_ch1']
    image_buffer_channel2 = features['image/encoded_ch2']

    if mtlclasses : 
        mtlabel = features['image/class/label_2']

    # Decode the jpeg
    with tf.name_scope(name='decode_png', values=[image_buffer_channel1]):
        image_channel1 = tf.image.decode_png(image_buffer_channel1, channels=1)
        image_channel1 = tf.image.convert_image_dtype(image_channel1, dtype=tf.float32)

    with tf.name_scope(name='decode_png', values=[image_buffer_channel2]):
        image_channel2 = tf.image.decode_png(image_buffer_channel2, channels=1)
        image_channel2 = tf.image.convert_image_dtype(image_channel2, dtype=tf.float32)

    if mtlclasses : 
        return label, image_channel1, image_channel2, mtlabel
    else :
        return label, image_channel1, image_channel2