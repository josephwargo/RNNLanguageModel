import numpy as np
class neuronLayer(object):
    def __init__(self, prevLayerShape, outputShape, activation, batchSize, clipVal, learningRate, rnn=False, adam=False):

        # layer info
        self.prevLayerShape = prevLayerShape
        self.outputShape = outputShape
        layerXavier = np.sqrt(2/(self.prevLayerShape+self.outputShape))
        self.layerWeights = np.random.normal(0,layerXavier, size=(self.prevLayerShape,self.outputShape)).astype(np.float32)
        if rnn:
            timeXavier = np.sqrt(2/(self.outputShape+self.outputShape))
            self.timeWeights = np.random.normal(0,timeXavier, size=(outputShape, outputShape)).astype(np.float32)
        self.bias = np.zeros(shape=(outputShape)).astype(np.float32)

        # activation
        self.activation = activation

        # hyperparameters
        self.batchSize = batchSize
        self.clipVal = clipVal
        self.learningRate = learningRate

        # storing hidden layer inputs and output during forward pass for BPTT
        self.prevLayerOutputMemory = []
        self.prevTimeStepOutputMemory = []
        self.thisLayerOutputMemory = []
        self.thisLayerMostRecentOutput = np.zeros(shape=(self.batchSize, outputShape)).astype(np.float32)

        # storing gradients during backward pass for BPTT
        self.thisLayerTimeLocalError = np.zeros(shape=(self.batchSize, outputShape)).astype(np.float32)
        if rnn:
            self.timeWeightUpdates = np.zeros_like(self.timeWeights)
        self.layerWeightUpdates = np.zeros_like(self.layerWeights)
        self.biasUpdates = np.zeros_like(self.bias)
        self.rnn = rnn

        # adam
        self.adam = adam
        if adam:
            # constants
            self.beta1 = .9
            self.beta2 = .999
            self.epsilon = 1e-8
            self.t = 1
            # arrays to store 
            self.mdLayerWeights = np.zeros(shape=(self.prevLayerShape,self.outputShape)).astype(np.float32)
            self.vdLayerWeights = np.zeros(shape=(self.prevLayerShape,self.outputShape)).astype(np.float32)
        
            if rnn:
                self.mdTimeWeights = np.zeros(shape=(self.outputShape,self.outputShape)).astype(np.float32)
                self.vdTimeWeights = np.zeros(shape=(self.outputShape,self.outputShape)).astype(np.float32)

            self.mdBias = np.zeros(shape=(outputShape)).astype(np.float32)         
            self.vdBias = np.zeros(shape=(outputShape)).astype(np.float32)
    
    def update(self, numSteps):

        # normalizing
        self.layerWeightUpdates /= numSteps
        if self.rnn:
            self.timeWeightUpdates /= numSteps
        self.biasUpdates /= numSteps

        # clipping
        np.clip(self.layerWeightUpdates, -self.clipVal, self.clipVal, out=self.layerWeightUpdates)
        if self.rnn:
            np.clip(self.timeWeightUpdates, -self.clipVal, self.clipVal, out=self.timeWeightUpdates)
        np.clip(self.biasUpdates, -self.clipVal, self.clipVal, out=self.biasUpdates)

        # adam
        if self.adam:
            self.updateAdam()
        
        else:
            self.layerWeights += -self.learningRate*self.layerWeightUpdates
            if self.rnn:
                self.timeWeights += -self.learningRate*self.timeWeightUpdates
            self.bias += -self.learningRate*self.biasUpdates


    def updateAdam(self):
        
        # doing ^t on beta1 and beta2 once per step
        b1T = self.beta1**self.t
        b2T = self.beta2**self.t
        
        # layer weights
        # momentum stored
        self.mdLayerWeights *= self.beta1
        self.mdLayerWeights += (1-self.beta1)*self.layerWeightUpdates
        # RMSProp stored
        self.vdLayerWeights *= self.beta2
        self.vdLayerWeights += (1-self.beta2)*(self.layerWeightUpdates**2)
        # updates (incl. corrections and Adam)
        self.layerWeights += -self.learningRate * (self.mdLayerWeights / (1-b1T) / (np.sqrt(self.vdLayerWeights / (1-b2T))+self.epsilon))

        if self.rnn:
            # time weights
            # momentum stored
            self.mdTimeWeights *= self.beta1
            self.mdTimeWeights += (1-self.beta1)*self.timeWeightUpdates
            # RMSProp stored
            self.vdTimeWeights *= self.beta2
            self.vdTimeWeights += (1-self.beta2)*(self.timeWeightUpdates**2)
            # update (incl. corrections and Adam)
            self.timeWeights += -self.learningRate * (self.mdTimeWeights / (1-b1T) / (np.sqrt(self.vdTimeWeights / (1-b2T))+self.epsilon))

        # bias
        # momentum stored
        self.mdBias *= self.beta1
        self.mdBias += (1-self.beta1)*self.biasUpdates
        # RMSProp stored
        self.vdBias *= self.beta2
        self.vdBias += (1-self.beta2)*(self.biasUpdates**2)
        # update (incl. corrections and Adam)
        self.bias += -self.learningRate * (self.mdBias / (1-b1T) / (np.sqrt(self.vdBias / (1-b2T))+self.epsilon))

        self.t+=1 # increment