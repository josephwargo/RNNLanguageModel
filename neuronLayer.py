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
            self.beta2 = .999
            self.epsilon = 10e-8
            self.t = 1
            # arrays to store 
            self.mdLayerWeights = np.zeros(shape=(self.prevLayerShape,self.outputShape))
            self.vdLayerWeights = np.zeros(shape=(self.prevLayerShape,self.outputShape))

            self.mdTimeWeights = np.zeros(shape=(self.outputShape,self.outputShape))
            self.vdTimeWeights = np.zeros(shape=(self.outputShape,self.outputShape))

            self.mdBias = np.zeros(shape=(outputShape))            
            self.vdBias = np.zeros(shape=(outputShape))
    
    def updateAdam(self, dLdLayerWeights, dLdTimeWeights, dLdBias):
        
        # doing ^t on beta1 and beta2 once per step
        b1T = self.beta1**self.t
        b2T = self.beta2**self.t

        # layer weights
        self.mdLayerWeights = self.beta1*self.mdLayerWeights + (1-self.beta1)*dLdLayerWeights # momentum stored
        mdLayerWeightsHat = self.mdLayerWeights / (1-b1T) # momentum correction
        self.vdLayerWeights = self.beta2*self.vdLayerWeights + (1-self.beta2)*(dLdLayerWeights**2) # RMSProp stored
        vdLayerWeightsHat = self.vdLayerWeights / (1-b2T) # RMSProp correction
        newdLdLayerWeights = mdLayerWeightsHat / (np.sqrt(vdLayerWeightsHat)+self.epsilon) # Adam

        # time weights
        self.mdTimeWeights = self.beta1*self.mdTimeWeights + (1-self.beta1)*dLdTimeWeights # momentum stored
        mdTimeWeightsHat = self.mdTimeWeights / (1-b1T) # momentum correction
        self.vdTimeWeights = self.beta2*self.vdTimeWeights + (1-self.beta2)*(dLdTimeWeights**2) # RMSProp stored
        vdTimeWeightsHat = self.vdTimeWeights / (1-b2T) # RMSProp correction
        newdLdTimeWeights = mdTimeWeightsHat / (np.sqrt(vdTimeWeightsHat)+self.epsilon) # Adam

        # bias
        self.mdBias = self.beta1*self.mdBias + (1-self.beta1)*dLdBias # momentum stored
        mdBiasHat = self.mdBias / (1-b1T) # momentum correction
        self.vdBias = self.beta2*self.vdBias + (1-self.beta2)*(dLdBias**2) # RMSprop stored
        vdBiasHat = self.vdBias / (1-b2T) # RMSProp correction
        newdLdBias = mdBiasHat / (np.sqrt(vdBiasHat)+self.epsilon) # Adam

        self.t+=1 # increment

        return newdLdLayerWeights, newdLdTimeWeights, newdLdBias