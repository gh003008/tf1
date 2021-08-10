
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
          
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]

y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X = tf.placeholder("float", [None,3]) # 입력 데이터 
Y = tf.placeholder("float", [None,3]) # 출력 데이터 계속 변하면서 넣어 줄수 있기 때문에 placeholder

W = tf.Variable(tf.random_normal([3,3])) # (X_attributes, Y_output)
b = tf.Variable(tf.random_normal([3]))

# weight 과 bias 를 이용한 후 softmax 함수 에 집어넣어 합이 1이 되는 출력으로 예측을 하는 모델을 만든다.
H = tf.nn.softmax(tf.matmul(X,W) + b) # hypothesis = logists = Y_hat 즉 예측값 

# cost/loss function = 우리의 예측값과 실제가 얼마나 차이 나는지 정도
# 이것을 cross entropy function을 사용해 본다.
# logistic cost(sigmoid의 응용)와 사실 같은 식이다!!
cost = tf.reduce_mean(-tf.reduce_sum(Y* tf.log(H), axis = 1)) 

# cost를 최소화 하는 방향으로 경사하강법을 이용하여 학습한다. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(H,1) # H를 최대로 만들어주는 입력값을 반환
is_correct = tf.equal(prediction, tf.argmax(Y,1)) # 실제와 예측값의 차이
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32)) # tf.cast: 텐서를 새로운 형태로 캐스팅 해줌 ex)int를 float로

# 설계한 모델 실행 및 그래프 생성
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)
################################ 이 부분까지 학습 ######################################################

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

#####COMMIT and push###