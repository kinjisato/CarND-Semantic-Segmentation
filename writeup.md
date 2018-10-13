## CarND-Semantic-Segmentation
### Kinji Sato / 13th October 2018
---

[//]: # (Image References)
[image1]: ./rubric/rubric01.png
[image2]: ./rubric/rubric02.png
[image3]: ./rubric/FCN.png
[image4]: ./runs/1539368453.6711168/um_000000.png
[image5]: ./runs/1539368453.6711168/um_000095.png
[image6]: ./runs/1539368453.6711168/umm_000000.png
[image7]: ./runs/1539368453.6711168/umm_000093.png
[image8]: ./runs/1539368453.6711168/uu_000000.png
[image9]: ./runs/1539368453.6711168/uu_000099.png

[video1]: ./runs/

## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### Build the Neural Network

![alt text][image1]

##### The function `load_vgg`

Alomot code was done in the lecure video. I added only remaing output layer3 layer4 and layer7.

```python
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Lecture video
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
```

##### The function `layers`

![alt text][image3]

`layer7_out` is FCN from `vgg_layer7-out`.

`lyaer4_in1` is upsample from `layer7_out`.

`layer4_in2` is FCN from `vgg_layer4_out`.

`layer4_out` is skip connection from `layer4_in1` and `layer4_in2`. `layer4_in1` and `layer4_in2` should be the same size.

`layer3_in1` is upsample frm `layer4_out`.

`layer3_in2` is FCN from `vgg_layer3_out`.

`layer3_out` is skip connection from `layer3_in1` and `layer3_in2`. `layer3_in1` and `layer3_in2` should be the same size.

`output` is upsample from `layer3_out`.  This layer would be returned.


```python
    layer7_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                    strides = (1,1),
                                    padding= 'same',
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # upsample
    layer4_in1 = tf.layers.conv2d_transpose(layer7_out, num_classes, 4,
                                            strides = (2,2),
                                            padding = 'same',
                                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 convolution of vgg layer 4
    layer4_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                    strides = (1,1),
                                    padding= 'same',
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # skip connection layer 4
    layer4_out = tf.add(layer4_in1, layer4_in2)

    # upsample
    layer3_in1 = tf.layers.conv2d_transpose(layer4_out, num_classes, 4,
                                            strides = (2,2),
                                            padding = 'same',
                                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # 1x1 convolution of vgg layer 3
    layer3_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                    strides = (1,1),
                                    padding= 'same',
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # skip connection layer 3
    layer3_out = tf.add(layer3_in1, layer3_in2)

    # upsample
    output = tf.layers.conv2d_transpose(layer3_out, num_classes, 16,
                                            strides = (8,8),
                                            padding = 'same',
                                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

```

##### The function `optimize`

`logits` and `correct_lavel` are reshaped to 2D.

`cross_entropy_loss` is come from the difference between `logits` and `correct_label`.

I choosed `Adam` optimizer. Maybe, this is reasonable choice.
(If there is much time to compare, those result should be compared.)

And our taget is to minimize the `cross_entropy_loss`


```python
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))

    # cross engropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    # optimizer Adam
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    # training
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
```

##### The function `train_nn`

Fisrt, I initialized variables with using `tf.global_variables_initializer()`.

And then NN would be trained with num of epochs and batche size defined.
To check `loss` is decreasing over the training process, `loss` is printed out at each batch loops.

```python
    # global initializer
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch+1))
        # placeholder
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                                feed_dict={
                                    input_image: image,
                                    correct_label: label,
                                    # placeholder
                                    keep_prob: 0.5,
                                    learning_rate: 0.001
                                })
            print("Loss: = {:.3f}".format(loss))

```


#### Neural Network Training

![alt text][image2]

##### Does the project train the model correctly?
##### On average, the model decreases loss over time.

Printed loss in the terminls shows loos was dereasssing over the training process.

##### Does the project use reasonable hyperparameters?
##### The number of epoch and batch size are set to a reasonable number.

When I ran my code on Amazon AMI EC2 instance that explaimed in the lecture, availabe bacth size was `2`. When I increased batch size more than 3, training process stopped due to the memory error. For Amazon AMI EC2 instance, I set batch size = 2, epochs = 100.
When I ran on Udacity workspace, batch sise = 5 was avaialbe (maybe more than 5 is ok to run), and I set ecpochs = 50. After 30 epochs training, loss value was alomost stable, so 30 - 50 epochs should be ok.

About `keep_prob` and `learning_rate`, I choosed 0.5 for `keep_prob` and 0.001 for `learning_rate`. With these valuses and batch size and num of epochs, result looks good. But, I'd like to have advise how I can find the best `keep_prob` and `learning_rate`. I don't have so much time and resource to try other values.

##### Does the project correctly label the road?

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

