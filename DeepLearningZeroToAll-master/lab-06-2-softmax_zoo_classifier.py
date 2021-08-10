# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
# for reproducibility # 아마 이부분은 tensorflow2.0에서는 오류가 발생 할 수도 있다. 
# 랜덤한 값을 다른 컴퓨터에서도 동일하게 얻을 수 있게 하는것.
# 랜덤이 사실 진짜 랜덤이 아님. 어디 적어논걸 순서대로 불러 오는건데 이 순서를 
# 다른 컴퓨터에서도 동일하게 불러오게 하면 랜덤하게 나오는 값들이 동일하게 나온다. 

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
''' 참고로 파일경로의 기준은 현재 터미널의 실행 위치가 기준임! 이 .py 실행 파일의 위치가 아님!!
    아래의 터미널의 위치를 보고 거기에 실행 파일을 두도록 한다. '''

x_data = xy[:, 0:-1] # 모든 행에 대해 처음~ 마지막 전 까지의 열 해당 데이터
y_data = xy[:, [-1]] # 모든 행에 대해 마지막 열 해당 데이터 (0~6)의 총 7개의 출력 (원핫처리 전)

print(x_data.shape, y_data.shape)

'''
(101, 16) (101, 1)
'''

nb_classes = 7  # 0 ~ 6 총 7개의 classification

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# one hot으로 만들때 one_hot함수를 쓰면 차원이 하나 더 늘어나게 되므로
# reshape를 사용하여 다시 차원은 하나 낮춰준, 우리가 원하는 형태의 one hot으로 
# 만들어 준다. 
# -1을 사용하는 것은 reshape의 알아서 맞춰주게 해주는 기능 
# ex) 2*6= 3*4

print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32) 
물음표는 x_data의 총 데이터 갯수와 같다. 
'''
 
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
# cost를 tf에서 지원하는 함수를 이용하여 간단하게 계산(softmax(logits)의 평균) 
# sigmoid/logistic cost와 cross entropy cost는 사실상 같은 것 \
'''
Logistic cost VS Cross entropy

- Cross entropy 에서는 A or B 로 [1,0] or [0,1] 로써 서로 구분을 하는거지만, 
  Logistic 은 정답이 A라고 했을 때 이게 맞냐 아니냐 , 혹은, B라고 했을때 이게 맞냐 아니냐를 판단하는것이다. 
- 따라서, 정답을 A라고 했을때 Ya = 1이기 때문에 Logistic에서 뒤에 텀은 날아가고 앞에 텀인 -log(H(x))만 남고, 
  이건 결국 Cross entropy에서 Li에 [1,0]를 넣으면 B의 확률이 0이라서 log값이 사라지기 때때문에 두 개의 식은 같은
  것이라고 볼 수 있다.
'''
# 여기서는 hypothesis를 사용하지도 않았음. 그 계산이 모두 포함 되어있다.    

# ENTROPY: 정보를 표현하는데 필요한 푀소 평균 자원량
# 몇비트로 표현할 수 있는가??
# 확률이 큰것은 길이가 짧고 확률이 작은 것은 길이가 길게 하는것이 효율적... sum[P * (-log(P))]
#                                                            
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# 얻어낸 cost를 W에 대해 미분하여 경사하강법으로 학습

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
                                        
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
...
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
'''
