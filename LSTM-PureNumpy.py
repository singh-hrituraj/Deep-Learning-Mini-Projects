import numpy as np
class RecurrentNeuralNetwork:
    
    def __init__(self, xs, ys, rl, eo, lr):
        
        self.x = np.zeros(x)
        self.xs = xs
        self.y = np.zeros(x)
        self.ys = ys
        self.w = np.random.random((ys, ys))
        
        self.G = np.zeros_like(self.w)
        
        self.rl = rl
        self.lr = lr
        
        self.ia = np.zeros((rl + 1, xs))
        self.ca = np.zeros((rl + 1, ys))
        self.oa = np.zeros((rl + 1, ys))
        self.ha = np.zeros((rl + 1, ys))
        self.af = np.zeros((rl + 1, ys))
        self.ai = np.zeros((rl + 1, ys))
        self.ac = np.zeros((rl + 1, ys))
        self.ao = np.zeros((rl + 1, ys))
        
        self.eo = np.zeros((rl + 1, ys))
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))
        self.LSTM = LSTM(xs, ys, rl, lr)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    def forward_prop(self):
        
        for i in range(1, self.rl+1):
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            
            cs, hs, f, c, o = self.LSTM.forward_prop()
            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i-1]
            
        return self.oa
    
      def backProp(self):
        
        totalError = 0
        
        dfcs = np.zeros(self.ys)
        dfhs = np.zeros(self.ys)
        #weight matrix
        tu = np.zeros((self.ys, self.ys))
        #LSTM Level Gradients
        tfu = np.zero((self.ys, self.xs + self.ys))
        tiu = np.zero((self.ys, self.xs + self.ys))
        tou = np.zeros((self.ys, self.xs + self.ys))
        
        
        for i in range(self.rl, -1, -1):
            error = self.oa[i] - self.eo[i]
            
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            
            error = np.dot(error, self.w)
            
            
            self.LSTM.x = np.hstack((self.hs[i-1], self.ia[i-1]))
            self.LSTM.cs = self.ca[i]
            #TO DO
            #complete backpropagation method and add update method
            
            
        
            
class LSTM:
    
    def __init__(self, xs, ys, rl, lr):
        self.x = np.zeros(xs + ys)
        self.xs = xs + ys
        self.y = np.zeros(ys)
        self.ys = ys
        self.cs = np.zeros(ys)
        self.rl = rl
        self.lr = lr
        
        
        self.f = np.random.random((ys, xs + ys))
        self.i = np.random.random((ys, xs + ys))
        self.c = np.random.random((ys, xs + ys))
        self.o = np.random.random((ys, xs + ys))
        
        self.Gf = np.zeros_like(self.f)
        self.Gi = np.zeros_like(self.i)
        self.Gc = np.zeros_like(self.c)
        self.Go = np.zeros_like(self.o)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tangent(self, x):
        return np.tanh(x)
    
    def dtangent(self, x):
        return 1 - np.tanh(x)**2
    
    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs+= i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        
        return self.cs, self.y, f, i, c, o
    
    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        
        e = np.clip(e + dfhs, -6, 6)
        do = self.tangent(self.cs) * e
        #Error with respect to O
        do = self.tangent(self.cs) * e
        
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        dc = dcs * i
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        di = dcs * c
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        df = dcs * pcs
        
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        dpcs = dcs * f
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df, self.f)[:self.ys]
        
        return fu, iu, cu, ou, dpcs, dphs
        
        #TO DO
        #Make the method implementable
        
        
        
    
            
        
        
