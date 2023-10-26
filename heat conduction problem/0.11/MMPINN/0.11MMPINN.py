#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import time
import math


# In[10]:


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
np.random.seed(1234)
tf.set_random_seed(1234)


# In[11]:


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, tb, X_f, layers, lb, ub,u_lb,u_ub):
        
        #    lb = np.array([-1, 0])      ub = np.array([1, 1])
        
        X0 = np.concatenate((x0, 0*x0+0.0), 1)              #    初始     
        X_lb = np.concatenate((0*tb + lb[0], tb), 1)    #    边界-1
        X_ub = np.concatenate((0*tb + ub[0], tb), 1)    #    边界+1    
        
        self.lb = lb
        self.ub = ub
               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]
        self.hsadasjd=1

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u_lb=u_lb
        self.u_ub=u_ub
        #分别是初始时刻的实部和虚部
        self.u0 = u0
        self.losslossloss=[]
        # Initialize NNs
        self.layers = layers
        #返回初始的权重w和偏差b
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders
        #形参 占位符，行数不确定，列数确定为1
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u_lb_tf = tf.placeholder(tf.float32, shape=[None, self.u_lb.shape[1]])
        self.u_ub_tf = tf.placeholder(tf.float32, shape=[None, self.u_ub.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs  进行预测
        self.u0_pred= self.net_uv(self.x0_tf, self.t0_tf)
        self.u_lb_pred= self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred= self.net_f_uv(self.x_f_tf, self.t_f_tf)
        
        # Loss   8个损失函数相加
        self.loss3=tf.pow(tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)),1/1)
        
        self.loss2=tf.pow(tf.reduce_mean(tf.square(self.f_u_pred)),1/1)
                
        
        self.loss = tf.pow(tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)),1)+                     tf.pow((tf.reduce_mean(tf.square(self.u_ub_tf  - self.u_ub_pred)) +                     tf.reduce_mean(tf.square(self.u_lb_tf  - self.u_lb_pred))),1) +                     tf.pow(tf.reduce_mean(tf.square(self.f_u_pred)),1/3)   
        self.lossss = tf.pow(tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)),1)+                     tf.pow((tf.reduce_mean(tf.square(self.u_ub_tf  - self.u_ub_pred)) +                     tf.reduce_mean(tf.square(self.u_lb_tf  - self.u_lb_pred))),1) +                     tf.pow(tf.reduce_mean(tf.square(self.f_u_pred)),1/2)  
        self.lossss33 = tf.pow(tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)),1)+                     tf.pow((tf.reduce_mean(tf.square(self.u_ub_tf  - self.u_ub_pred)) +                     tf.reduce_mean(tf.square(self.u_lb_tf  - self.u_lb_pred))),1) +                     tf.pow(tf.reduce_mean(tf.square(self.f_u_pred)),1)  
                        
        
        
        
        
        
        
        
        self.loss4 = tf.pow((tf.reduce_mean(tf.square(self.u_ub_tf  - self.u_ub_pred)) +                     tf.reduce_mean(tf.square(self.u_lb_tf  - self.u_lb_pred))),1/1)
        
        
        # Optimizers  maxiter最大迭代次数  maxfun最大求值次数 maxcor int变量的最大数量
        #maxls 可选的最大搜索步数
        #maxls 可选的最大搜索步数
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 100000,
                                                                           'maxfun': 100000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        
        
        self.optimizer11 = tf.contrib.opt.ScipyOptimizerInterface(self.lossss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 100000,
                                                                           'maxfun': 100000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        
        self.optimizer22 = tf.contrib.opt.ScipyOptimizerInterface(self.lossss33, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 100000,
                                                                           'maxfun': 100000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
                
        
        
        
        
        
        
        
        
        '''
        是一个寻找全局最优点的优化算法，引入了二次方梯度校正
        除了利用反向传播算法对权重和偏置项进行修正外，也在运行中不断修正学习率。
        根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，损失量小，修正的幅度也小，学习率就小，
        但是不会超过自己所设定的学习率。20
        3
        '''
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session  配置Session运行参数&&GPU设备指定）
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        #初始化模型的参数
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        #产生截断正态分布随机数，stddev是标准差，取值范围为[ 0 - 2 * stddev, 0+2 * stddev ]
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        #将初始输入X映射到-1到1之间为H
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x, t):
        X = tf.concat([x,t],1)
        
        uv = self.neural_net(X, self.weights, self.biases)


        return uv
    
    
    
    def net_f_uv(self, x, t):
        
        u = self.net_uv(x,t)       
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        hhhhhh=0.11
        f_u = u_t-u_xx-2*tf.exp(1/((2*t-1)*(2*t-1)+hhhhhh))+tf.exp(1/((2*t-1)*(2*t-1)+hhhhhh))*(1-x*x)*4*(2*t-1)/((2*t-1)*(2*t-1)+hhhhhh)/((2*t-1)*(2*t-1)+hhhhhh)
        #f_u=u-tf.exp(1/((2*t-1)*(2*t-1)+0.5))*(1-x*x)
        #return f_u/1319.919299519142
        return f_u
    
    def callback(self, loss,f_u_pred,u0_pred,u_ub_pred,u_lb_pred):
        
        self.losslossloss.append(loss)
            #losslossloss2
        sss=self.hsadasjd
        if sss%20==0:
            losssss =tf.reduce_mean(tf.square(f_u_pred))
            array1 = losssss.eval(session=tf.Session())        
            tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,self.u_lb_tf:self.u_lb,self.u_ub_tf:self.u_ub,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
            loss1123456=self.u0_tf
            lossskdajsdkas=self.sess.run(loss1123456, tf_dict)
            zkjxJXhz = tf.reduce_mean(tf.square(lossskdajsdkas - u0_pred))
            array2 = zkjxJXhz.eval(session=tf.Session())
        
            loss1123456=self.u_ub_tf
            lssss1=self.sess.run(loss1123456, tf_dict)
            loss112345sds6=self.u_lb_tf
            sadsk=self.sess.run(loss112345sds6, tf_dict)            
            
            zkjxJXhzs = tf.reduce_mean(tf.square( lssss1- u_ub_pred))+tf.reduce_mean(tf.square(sadsk  - u_lb_pred))
            array4 = zkjxJXhzs.eval(session=tf.Session())
            print('It: %d, Loss1: %.9e,loss2: %.9e Loss3: %.9e' % 
                      (sss,array2,array4,array1))
            log5=open("log5.txt",mode = 'a+', encoding = 'utf-8')
            print(array2,file=log5)
            log5.close() 
            log6=open("log6.txt",mode = 'a+', encoding = 'utf-8')
            print(array4,file=log6)
            log6.close()            
            log7=open("log7.txt",mode = 'a+', encoding = 'utf-8')
            print(array1,file=log7)
            log7.close()
        sss=sss+1
        self.hsadasjd=sss
    
    
    def callback11(self, lossss,f_u_pred,u0_pred,u_ub_pred,u_lb_pred):
        
        self.losslossloss.append(lossss)
            #losslossloss2
        sss=self.hsadasjd
        if sss%20==0:
            losssss =tf.reduce_mean(tf.square(f_u_pred))
            array1 = losssss.eval(session=tf.Session())        
            tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,self.u_lb_tf:self.u_lb,self.u_ub_tf:self.u_ub,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
            loss1123456=self.u0_tf
            lossskdajsdkas=self.sess.run(loss1123456, tf_dict)
            zkjxJXhz = tf.reduce_mean(tf.square(lossskdajsdkas - u0_pred))
            array2 = zkjxJXhz.eval(session=tf.Session())
        
            loss1123456=self.u_ub_tf
            lssss1=self.sess.run(loss1123456, tf_dict)
            loss112345sds6=self.u_lb_tf
            sadsk=self.sess.run(loss112345sds6, tf_dict)            
            
            zkjxJXhzs = tf.reduce_mean(tf.square( lssss1- u_ub_pred))+tf.reduce_mean(tf.square(sadsk  - u_lb_pred))
            array4 = zkjxJXhzs.eval(session=tf.Session())
            print('It: %d, Loss1: %.9e,loss2: %.9e Loss3: %.9e' % 
                      (sss,array2,array4,array1))
        sss=sss+1
        self.hsadasjd=sss    
    
    
    def callback22(self, lossss33,f_u_pred,u0_pred,u_ub_pred,u_lb_pred):
        
        self.losslossloss.append(lossss33)
            #losslossloss2
        sss=self.hsadasjd
        if sss%20==0:
            losssss =tf.reduce_mean(tf.square(f_u_pred))
            array1 = losssss.eval(session=tf.Session())        
            tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,self.u_lb_tf:self.u_lb,self.u_ub_tf:self.u_ub,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
            loss1123456=self.u0_tf
            lossskdajsdkas=self.sess.run(loss1123456, tf_dict)
            zkjxJXhz = tf.reduce_mean(tf.square(lossskdajsdkas - u0_pred))
            array2 = zkjxJXhz.eval(session=tf.Session())
        
            loss1123456=self.u_ub_tf
            lssss1=self.sess.run(loss1123456, tf_dict)
            loss112345sds6=self.u_lb_tf
            sadsk=self.sess.run(loss112345sds6, tf_dict)            
            
            zkjxJXhzs = tf.reduce_mean(tf.square( lssss1- u_ub_pred))+tf.reduce_mean(tf.square(sadsk  - u_lb_pred))
            array4 = zkjxJXhzs.eval(session=tf.Session())
            print('It: %d, Loss1: %.9e,loss2: %.9e Loss3: %.9e' % 
                      (sss,array2,array4,array1))
        sss=sss+1
        self.hsadasjd=sss 
        
    
    
    
    
    
    
    def train(self, nIter):   
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,self.u_lb_tf:self.u_lb,self.u_ub_tf:self.u_ub,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        lossloss1 = []
        lossloss2 = []
        lossloss3=[]
        
        start_time = time.time()
        
        loss_value11 = self.sess.run(self.loss3, tf_dict)
        lossloss1.append(loss_value11)
        
        loss_value22 = self.sess.run(self.loss2, tf_dict)
        lossloss2.append(loss_value22)
        
        loss_value33 = self.sess.run(self.loss4, tf_dict)
        lossloss3.append(loss_value33)
        
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if it % 20 == 0:
                elapsed = time.time() - start_time
                
                
                loss_value11 = self.sess.run(self.loss3, tf_dict)
                lossloss1.append(loss_value11)
                
                loss_value22 = self.sess.run(self.loss2, tf_dict)
                lossloss2.append(loss_value22)
                
                loss_value33 = self.sess.run(self.loss4, tf_dict)
                lossloss3.append(loss_value33)
                
                print('It: %d, Loss1: %.9e,loss2: %.9e Loss3: %.9e,Time: %.2f' % 
                      (it, loss_value11,loss_value33,loss_value22, elapsed))
                start_time = time.time()
                log1=open("log1.txt",mode = 'a+', encoding = 'utf-8')
                print(loss_value11,file=log1)
                log1.close()
                log2=open("log2.txt",mode = 'a+', encoding = 'utf-8')
                print(loss_value33,file=log2)
                log2.close()
                log3=open("log3.txt",mode = 'a+', encoding = 'utf-8')
                print(loss_value22,file=log3)
                log3.close()
                
                

        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict, 
                                fetches = [self.loss,self.f_u_pred,self.u0_pred,self.u_ub_pred,self.u_lb_pred], 
                                loss_callback = self.callback
                               )    
        
        
        
        lb = np.array([-1, 0])
        ub = np.array([1, 1])

        x=np.linspace(-1,1,4000).flatten()[:,None]   
        t=np.linspace(0,1,4000).flatten()[:,None]  
        X, T = np.meshgrid(x, t)
                
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_pred, f_u_pred = model.predict(X_star)

        scipy.io.savemat("1.mat", {'u': u_pred})     
        
    
        self.optimizer11.minimize(self.sess, 
                                feed_dict = tf_dict, 
                                fetches = [self.lossss,self.f_u_pred,self.u0_pred,self.u_ub_pred,self.u_lb_pred], 
                                loss_callback = self.callback11
                               )  
        
        
        
        lb = np.array([-1, 0])
        ub = np.array([1, 1])

        x=np.linspace(-1,1,4000).flatten()[:,None]   
        t=np.linspace(0,1,4000).flatten()[:,None]  
        X, T = np.meshgrid(x, t)
                
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_pred, f_u_pred = model.predict(X_star)
        scipy.io.savemat("2.mat", {'u': u_pred})         

        
        
        
        
        self.optimizer22.minimize(self.sess, 
                                feed_dict = tf_dict, 
                                fetches = [self.lossss33,self.f_u_pred,self.u0_pred,self.u_ub_pred,self.u_lb_pred], 
                                loss_callback = self.callback22
                               )         
        lb = np.array([-1, 0])
        ub = np.array([1, 1])

        x=np.linspace(-1,1,4000).flatten()[:,None]   
        t=np.linspace(0,1,4000).flatten()[:,None]  
        X, T = np.meshgrid(x, t)
                
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_pred, f_u_pred = model.predict(X_star)
        
        scipy.io.savemat("3.mat", {'u': u_pred})
        
        
        
        
        return lossloss1,lossloss2
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)  
        
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
               
        return u_star,f_u_star
    def loss_show(self):
        return self.losslossloss


# In[12]:


def heatsolution(x,t):
    return math.exp(1/((2*t-1)**2+0.11))*(1-x**2)


# In[13]:


if __name__ == "__main__": 
         
    
    # Doman bounds
    lb = np.array([-1, 0])
    ub = np.array([1, 1])

    N0 = 4000                                      #初始点
    N_b = 4000                                     #边界点
    N_f = 100000                              #适配点
    layers = [2,400,400,400,400,1]  
    #读取真实解
    x=np.linspace(-1,1,4000).flatten()[:,None]   
    t=np.linspace(0,1,4000).flatten()[:,None]   
    res=np.zeros([len(x),len(t)])  
    for i in range(len(x)):
        for j in range(len(t)):
            res[i,j]=heatsolution(x[i],t[j])
    
    
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    #选定初始点N0=700个点
    idx_x = np.random.choice(x.shape[0], N0, replace=False)   
    x0 = x[idx_x,:]
    u0 = res[idx_x,0:1]
    #选择N_b=700个边界点
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    u_lb = res[0,idx_t]
    u_ub=res[-1,idx_t]
    #N_f=2500个随机搭配点   第一列位置 第二列时间
    X_f = lb + (ub-lb)*lhs(2, N_f)
    x0=np.array(x0).flatten()[:,None]
    u0=np.array(u0).flatten()[:,None]
    u_lb=np.array(u_lb).flatten()[:,None]
    u_ub=np.array(u_ub).flatten()[:,None]


# In[14]:


model = PhysicsInformedNN(x0, u0,tb, X_f, layers, lb, ub,u_lb,u_ub)     


# In[15]:


LOSS1,LOSS2=model.train(2000)


# In[ ]:


X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_pred, f_u_pred = model.predict(X_star)
u_star = res.T.flatten()[:,None]  
error_u1 = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
error_u2 = np.linalg.norm(u_star-u_pred,1)/len(u_star)
error_u3 = np.linalg.norm(u_star-u_pred,np.inf)
print('二范数Error u: %e' % (error_u1))
print('平均绝对Error u: %e' % (error_u2))
print('无穷范数Error u: %e' % (error_u3))


# In[ ]:




