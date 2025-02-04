import numpy as np
class neuronLayer(object):
    def __init__(self, prevLayerShape, outputShape, rnn=False, adam=False):
        self.prevLayerShape = prevLayerShape
        self.outputShape = outputShape
        xavier = np.sqrt(2/(self.prevLayerShape+self.outputShape))
        self.layerW = np.random.normal(0,xavier, size=(self.prevLayerShape,self.outputShape))
        if rnn:
            self.timeW = np.zeros(shape=(outputShape, outputShape))
        self.b = np.zeros(shape=(outputShape))
        self.N = np.zeros(shape=(outputShape))
        self.NMemory = []
        
        # adam
        self.adam = adam
        if adam:
            # constants
            self.beta1 = .9
            self.beta1T = self.beta1
            self.beta2 = .999
            self.beta2T = self.beta2
            self.epsilon = 10e-8
            self.t = 1
            # arrays to store 
            self.mdW = np.zeros(shape=(self.prevLayerShape,self.outputShape))
            self.mdB = np.zeros(shape=(outputShape))
            self.vdW = np.zeros(shape=(self.prevLayerShape,self.outputShape))
            self.vdB = np.zeros(shape=(outputShape))
    
    def updateAdam(self, dCdW, dCdB):
        # momentum
        self.mdW = self.beta1*self.mdW + (1-self.beta1)*dCdW
        self.mdB = self.beta1*self.mdB + (1-self.beta1)*dCdB
        # RMSprop
        self.vdW = self.beta2*self.vdW + (1-self.beta2)*(dCdW**2)
        self.vdB = self.beta2*self.vdB + (1-self.beta2)*(dCdB**2)
        # bias correction
        mdWHat = self.mdW / (1-self.beta1**self.t)
        mdBHat = self.mdB / (1-self.beta1**self.t)
        vdWHat = self.vdW / (1-self.beta2**self.t)
        vdBHat = self.vdB / (1-self.beta2**self.t)
        # adam
        newdCdW = mdWHat / (np.sqrt(vdWHat)+self.epsilon)
        newdCdB = mdBHat / (np.sqrt(vdBHat)+self.epsilon)
        self.t+=1
        return newdCdW, newdCdB