import numpy as np
from neuronLayer import neuronLayer
import costsAndActivations as caa

# entire net
class neuralNet(object):
    def __init__(self, embeddings, word2ind, outputActivation, hiddenLayerShapes, 
                 hiddenLayerActivations, lossFunction='crossEntropyLoss', learningRate=.001, epochs=1,
                 adam=False, clipVal=1, debug=False):
        # errors
        if len(hiddenLayerShapes)!=len(hiddenLayerActivations):
            raise Exception('Length of hiddenLayerShapes does not match length of hiddenLayerActivations')
        if ((lossFunction=='crossEntropyLoss') & (outputActivation!='softmax')) or ((lossFunction!='crossEntropyLoss') & (outputActivation=='softmax')):
            raise Exception('A cost function of Cross Entropy Loss and an output layer activation of Softmax must be paired with each other')
        if adam & (learningRate>.01):
            print('Warning: Learning rate may be too high for ADAM optimizer to function properly')
        # variables straight from initialization
        # embeddings
        self.embeddings = embeddings
        # self.corpus = corpus
        self.word2ind = word2ind
        self.embeddingsShape = embeddings.shape[1]
        self.numEmbeddings = embeddings.shape[0]
        
        # hyperparameters
        self.epochs = epochs
        self.debug = debug
        self.adam = adam
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        self.clipVal = clipVal
        self.activations = hiddenLayerActivations + [outputActivation]

        # loss
        self.localLoss = []
        self.loss = []
        self.lossGradients = []
        
        # initializing hidden layers and adding to dictionary of all layers
        hiddenLayer1 = neuronLayer(self.embeddingsShape, hiddenLayerShapes[0],
                                   hiddenLayerActivations[0], rnn=True, adam=adam)
        self.layers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, inputShape in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.layers["hiddenLayer{}".format(layerNum)] = neuronLayer(
                        inputShape, hiddenLayerShapes[count+1], hiddenLayerActivations[count+1], rnn=True, adam=adam)
       
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(
            prevLayerShape=hiddenLayerShapes[-1], outputShape=self.numEmbeddings, activation=outputActivation, rnn=True, adam=adam)
        self.layers['outputLayer'] = outputLayer


    def forwardPassPerWord(self, prevLayerOutput, train):
        # cycling through each layer
        for layerName in self.layers.keys():
            currLayer = self.layers[layerName]
            # for output layer
            if layerName == 'outputLayer':
                if train:
                    # adding previous layer hidden state to memory (for BPTT)
                    currLayer.prevLayerOutputMemory.append(prevLayerOutput)

                z = np.dot(prevLayerOutput, currLayer.layerWeights) + currLayer.bias
                logits = z
                
                # updating hidden state
                currLayer.thisLayerMostRecentOutput = prevLayerOutput

            # for non-output layers
            else:
                if train:
                    # adding previous layer and timestep hidden states to memory (for BPTT)
                    currLayer.prevLayerOutputMemory.append(prevLayerOutput)
                    currLayer.prevTimeStepOutputMemory.append(currLayer.thisLayerMostRecentOutput)
                
                # calculating hidden state (dot products and activation)
                layerDotProduct = np.dot(prevLayerOutput, currLayer.layerWeights)
                timeDotProduct = np.dot(currLayer.thisLayerMostRecentOutput, currLayer.timeWeights)
                z = layerDotProduct + timeDotProduct + currLayer.bias # z = Uh + Wx + b
                hiddenState = caa.activation(currLayer.activation, z) # activation - depending on the layer

                # updating hidden state and hidden state memory
                currLayer.thisLayerMostRecentOutput = hiddenState
                if train:
                    currLayer.thisLayerOutputMemory.append(hiddenState)

                # updating so that this outputs hidden state feeds into 
                prevLayerOutput = hiddenState
        return logits

    # training methods
    def forwardPass(self, text, train=True):
        # cycling through each word (timestep)
        for wordIndex in range(len(text)-1):
            # selecting proper input embeddings
            inputWord = text[wordIndex]
            inputVocabIndex = self.word2ind[inputWord]
            prevLayerOutput = self.embeddings[inputVocabIndex]

            # selecting index for output word
            outputWord = text[wordIndex+1]
            outputVocabIndex = self.word2ind[outputWord]
            
            # iterating through each layer per word
            logits = self.forwardPassPerWord(prevLayerOutput, train)

            if train:
                # storing local loss and loss gradients
                if self.lossFunction == 'crossEntropyLoss':
                    loss = caa.crossEntropyLoss(outputVocabIndex, logits)
                    self.localLoss.append(loss)
                    self.lossGradients.append(caa.softmaxLocalError(outputVocabIndex, logits)) # dLdH
                else:
                    raise Exception('Unknown loss function')
        if train:
            # calculating final loss (divided over all words in the text)
            meanLoss = np.mean(self.localLoss)
            self.loss.append(meanLoss)
        
        return logits

    def backwardPassPerTimestep(self, layerLocalError, reverseKeys, timeStep):
        # iterating through layers backwards
        for layerNum, layerName in enumerate(reverseKeys):
            layerNum = len(reverseKeys) - 1 - layerNum
            currLayer = self.layers[layerName]
            
            if layerName == 'outputLayer':
                # dLossdZ [stored during forward pass]
                dLossdZ = layerLocalError

                # dLossdOutputWeights [dLdW] = dZdW [previous layer hidden state output] @ dLdZ [stored during forward pass]
                prevLayerHiddenState = currLayer.prevLayerOutputMemory[timeStep]
                dLossdOutputWeights = np.outer(prevLayerHiddenState, dLossdZ)
                
                # dLossdOutputBias [dLdB] = dLdZ [stored during forward pass] @ dZdB [1]
                dLossdOutputBias = layerLocalError

                # dLossdPreviousHiddenLayer [dLdH] = dZdH [layerWeights] @ dLdZ [stored during forward pass]
                dLossdPrevLayerHiddenState = np.dot(currLayer.layerWeights, layerLocalError)
                
                # updating localError to pass back
                layerLocalError = dLossdPrevLayerHiddenState

                # adam
                # if currLayer.adam:
                #     dLossdOutputWeights, _, dLossdOutputBias = currLayer.updateAdam(dLossdOutputWeights, 0, dLossdOutputBias)

                # adding gradients to list for weight updates
                currLayer.layerWeightUpdates += dLossdOutputWeights
                currLayer.biasUpdates += dLossdOutputBias
            
            else:
                # dLossdZ [dLdZ] = localError(dLdH [passed back from previous layer / time step], dHdZ [this layer hidden state most recent output])
                dLdH = layerLocalError+currLayer.thisLayerTimeLocalError
                hiddenState = currLayer.thisLayerOutputMemory[timeStep]
                dLossdZ = caa.localError(currLayer.activation, hiddenState, dLdH)
                
                # dLossdTimeWeights [dLdWt] = dLdZ [calculated above] @ dZdWt [prevTimeStepHiddenState]
                prevTimeStepHiddenState = currLayer.prevTimeStepOutputMemory[timeStep]
                dLossdTimeWeights = np.outer(prevTimeStepHiddenState, dLossdZ)
                
                # dLossdLayerWeights [dLdWl] = dZdWl [prevLayerHiddenState] @ dLdZ [calculated above]
                prevLayerHiddenState = currLayer.prevLayerOutputMemory[timeStep]
                dLossdLayerWeights = np.outer(prevLayerHiddenState, dLossdZ)
                
                # dLossdOutputBias [dLdB] = dLdZ [calculated above] @ dZdB = [1]
                dLossdOutputBias = dLossdZ

                # dLossdPreviousHiddenLayer [dLdH] = dZdH [layerWeights] @ dLdZ [calculated above]
                layerLocalError = np.dot(currLayer.layerWeights, dLossdZ)
                
                # dLossdPreviousTimeStep [dLdH] = dZdH [timeWeights] @ dLdZ [calculated above]
                currLayer.thisLayerTimeLocalError = np.dot(currLayer.timeWeights, dLossdZ)
                
                # adding gradients to list for weight updates
                currLayer.layerWeightUpdates += dLossdLayerWeights
                currLayer.timeWeightUpdates += dLossdTimeWeights
                currLayer.biasUpdates += dLossdOutputBias

    # backward pass through entire text
    def backwardPass(self, text):
        numSteps = len(text)
        # reversing layers to iterate backwards through
        reverseKeys = list(self.layers.keys())
        reverseKeys.reverse()

        # iterating through time backwards
        for reverseTimeStep in range(1, numSteps):
            timeStep = numSteps - reverseTimeStep - 1
            
            # getting proper local error that was stored during forward pass
            layerLocalError = self.lossGradients[timeStep]

            # NOT RELEVANT UNLESS WE WANT TO UPDATE EMBEDDINGS selecting proper input embeddings
            # inputWord = text[timeStep]
            # inputVocabIndex = self.word2ind[inputWord]
            # inputWordEmbedding = self.embeddings[inputVocabIndex]

            # backward pass for an individual time step
            self.backwardPassPerTimestep(layerLocalError, reverseKeys, timeStep)
        
        # updating weights and biases
        for layerName in reverseKeys:
            currLayer = self.layers[layerName]

            # normalize
            currLayer.layerWeightUpdates /= (numSteps-1)
            if layerName != 'outputLayer':
                currLayer.timeWeightUpdates /= (numSteps-1)
            currLayer.biasUpdates /= (numSteps-1)
            
            # clipping
            currLayer.layerWeightUpdates = np.clip(currLayer.layerWeightUpdates, -self.clipVal, self.clipVal)
            if layerName != 'outputLayer':
                currLayer.timeWeightUpdates = np.clip(currLayer.timeWeightUpdates, -self.clipVal, self.clipVal)
            currLayer.biasUpdates = np.clip(currLayer.biasUpdates, -self.clipVal, self.clipVal)

            # adam
            if currLayer.adam:
                currLayer.layerWeightUpdates, currLayer.timeWeightUpdates, currLayer.biasUpdates = currLayer.updateAdam(
                    currLayer.layerWeightUpdates, currLayer.timeWeightUpdates, currLayer.biasUpdates)

            # updates
            currLayer.layerWeights += -self.learningRate*currLayer.layerWeightUpdates
            if layerName != 'outputLayer':
                currLayer.timeWeights += -self.learningRate*currLayer.timeWeightUpdates
            currLayer.bias += -self.learningRate*currLayer.biasUpdates
    
    def resetGrads(self):
    # Resetting losses and gradients in advance of forward pass
        # setting/resetting loss
        self.localLoss = [] # list to store the local loss to average at the end
        self.lossGradients = [] # list used to store loss gradients for backwards pass
                                # resetting list to empty so we only store gradients from this
                                # instance of the forward pass
        
        # setting/resetting gradients & other stored variables per layer
        for layerName in self.layers.keys():
            currLayer = self.layers[layerName] # layer

            # for forward pass calculation
            currLayer.thisLayerMostRecentOutput = np.zeros(currLayer.thisLayerMostRecentOutput.shape) # hidden state at time 0
            
            # for backward pass calculation
            currLayer.prevLayerOutputMemory = [] # memory of hidden states from the previous layer in  this timestep
            currLayer.prevTimeStepOutputMemory = [] # memory of hidden states from this layer in the previous timestep
            currLayer.thisLayerOutputMemory = [] # memory of the output from this layer

            # for gradient updates
            currLayer.thisLayerTimeLocalError = np.zeros(shape=(currLayer.thisLayerTimeLocalError.shape))
            currLayer.timeWeightUpdates = 0
            currLayer.layerWeightUpdates = 0
            currLayer.biasUpdates = 0

    # training model by repeatedly running forward and backward passes
    def trainModel(self, corpus):
        for count, text in enumerate(corpus):
            wordCount = len(text)
            print(f'Text #{count+1} - {wordCount} words')
            
            # resetting gradients
            self.resetGrads()

            # forward pass
            _ = self.forwardPass(text)
            modelLoss = self.loss[-1]
            print(f'Loss: {modelLoss}')
            print('********************************************')
            print()

            # backward pass
            self.backwardPass(text)

    # return predicted output for a given input
    def querySequence(self, sequence):

        # getting logits
        logits = self.forwardPass(sequence, train=False)
        
        # determining output word
        outputWordIndex = np.argmax(logits)
        outputWord = list(self.word2ind.keys())[outputWordIndex]

        return outputWord
    

    def generateOutput(self, sequence, numWords):
        
        for i in range(numWords):
            nextWord = self.querySequence(sequence)
            sequence.append(nextWord)
        
        return sequence