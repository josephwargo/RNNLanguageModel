import numpy as np
class neuronLayer(object):
    def __init__(self, prevLayerShape, outputShape, activation, rnn=False, adam=False):

        # layer info
        self.prevLayerShape = prevLayerShape
        self.outputShape = outputShape
        xavier = np.sqrt(2/(self.prevLayerShape+self.outputShape))
        self.layerWeights = np.random.normal(0,xavier, size=(self.prevLayerShape,self.outputShape))
        if rnn:
            self.timeWeights = np.zeros(shape=(outputShape, outputShape))
        self.bias = np.zeros(shape=(outputShape))

        # activation
        self.activation = activation

        # storing hidden layer inputs and output during forward pass for BPTT
        self.prevLayerOutputMemory = []
        self.prevTimeStepOutputMemory = []
        self.thisLayerOutputMemory = []
        self.thisLayerMostRecentOutput = np.zeros(shape=(outputShape))

        # storing gradients during backward pass for BPTT
        self.thisLayerTimeLocalError = np.zeros(shape=(outputShape))
        self.timeWeightUpdates = 0
        self.layerWeightUpdates = 0
        self.biasUpdates = 0

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