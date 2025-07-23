import numpy as np
from neuronLayer import neuronLayer
import costsAndActivations as caa

# entire net
class neuralNet(object):
    def __init__(self, embeddings, word2ind, outputActivation, hiddenLayerShapes, 
                 hiddenLayerActivations, lossFunction='crossEntropyLoss', learningRate=.001, epochs=1, batchSize=8,
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
        self.word2indMapper = np.vectorize(word2ind.get)
        self.embeddingsShape = embeddings.shape[1]
        self.numEmbeddings = embeddings.shape[0]
        
        # hyperparameters
        self.epochs = epochs
        self.batchSize = batchSize
        self.debug = debug
        self.adam = adam
        self.learningRate = np.float32(learningRate)
        self.lossFunction = lossFunction
        self.clipVal = np.float32(clipVal)
        self.activations = hiddenLayerActivations + [outputActivation]

        # loss
        self.localLoss = []
        self.loss = []
        self.lossGradients = []
        
        # initializing hidden layers and adding to dictionary of all layers
        hiddenLayer1 = neuronLayer(self.embeddingsShape, hiddenLayerShapes[0],
                                    hiddenLayerActivations[0], batchSize=self.batchSize,
                                    clipVal=self.clipVal, learningRate=self.learningRate,
                                    rnn=True, adam=adam)
        self.layers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, inputShape in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.layers["hiddenLayer{}".format(layerNum)] = neuronLayer(
                        inputShape, hiddenLayerShapes[count+1], hiddenLayerActivations[count+1],
                        batchSize=self.batchSize, clipVal=self.clipVal, learningRate=self.learningRate,
                        rnn=True, adam=adam)
       
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(
            prevLayerShape=hiddenLayerShapes[-1], outputShape=self.numEmbeddings, activation=outputActivation,
            batchSize=self.batchSize, clipVal=self.clipVal, learningRate=self.learningRate,
            rnn=False, adam=adam)
        self.layers['outputLayer'] = outputLayer

        self.currentText = None
        self.padIndex = self.word2ind['<PAD>']

    def inputToInd(self, text, train=True):
        if train:
            longestSequence = max(len(sequence) for sequence in text)
            padVal = '<PAD>'
            text = [sequence + [padVal] * (longestSequence - len(sequence)) for sequence in text]
        text = self.word2indMapper(text)

        return text

    def forwardPassPerWord(self, prevLayerOutput, train):
        # cycling through each layer
        for layerName in self.layers.keys():
            currLayer = self.layers[layerName]
            # for output layer
            if layerName == 'outputLayer':
                if train:
                    # adding previous layer hidden state to memory (for BPTT)
                    currLayer.prevLayerOutputMemory.append(prevLayerOutput)

                z = prevLayerOutput @ currLayer.layerWeights + currLayer.bias
                logits = z
                # print(logits.shape)
                # updating hidden state
                currLayer.thisLayerMostRecentOutput = prevLayerOutput

            # for non-output layers
            else:
                if train:
                    # adding previous layer and timestep hidden states to memory (for BPTT)
                    currLayer.prevLayerOutputMemory.append(prevLayerOutput)
                    currLayer.prevTimeStepOutputMemory.append(currLayer.thisLayerMostRecentOutput.copy())
                
                # calculating hidden state (dot products and activation)
                # layerDotProduct = np.dot(prevLayerOutput, currLayer.layerWeights)
                layerDotProduct = prevLayerOutput @ currLayer.layerWeights
                # print(layerDotProduct.shape)
                timeDotProduct = currLayer.thisLayerMostRecentOutput @ currLayer.timeWeights
                # print(timeDotProduct.shape)
                # print(timeDotProduct.shape)
                z = layerDotProduct + timeDotProduct + currLayer.bias # z = Uh + Wx + b
                hiddenState = caa.activation(currLayer.activation, z) # activation - depending on the layer
                # print(hiddenState.shape)
                # updating hidden state and hidden state memory
                currLayer.thisLayerMostRecentOutput = hiddenState
                if train:
                    currLayer.thisLayerOutputMemory.append(hiddenState.copy())

                # updating so that this outputs hidden state feeds into 
                prevLayerOutput = hiddenState
        return logits

    # training methods
    def forwardPass(self, text, train=True):
        # cycling through each word (timestep)
        self.currentText = self.inputToInd(text, train)
        if train:
            numSteps = self.currentText.shape[1]
        else:
            numSteps = self.currentText.shape[0]
        for wordIndex in range(numSteps-1):
            # selecting proper input embeddings
            if train:
                inputWords = self.currentText[:,wordIndex]
            else:
                inputWords = self.currentText[wordIndex]

            prevLayerOutput = self.embeddings[inputWords]
            

            # selecting index for output word
            if train:
                outputVocabIndex = self.currentText[:, wordIndex+1]
                padMask = outputVocabIndex!=self.padIndex
            else:
                outputVocabIndex = self.currentText[wordIndex+1]
            # outputVocabIndex = self.word2ind[outputWord]
            
            # iterating through each layer per word
            logits = self.forwardPassPerWord(prevLayerOutput, train)

            if train:
                # storing local loss and loss gradients
                if self.lossFunction == 'crossEntropyLoss':
                    loss = caa.crossEntropyLoss(outputVocabIndex[padMask], logits[padMask])
                    self.localLoss.append(loss)
                    # gradient
                    gradient = caa.softmaxLocalError(outputVocabIndex[padMask], logits[padMask])
                    fullGradient = np.zeros_like(logits)
                    fullGradient[padMask] = gradient
                    self.lossGradients.append(fullGradient) # dLdH
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
                dLossdOutputWeights = prevLayerHiddenState.T @ dLossdZ
                # dLossdOutputWeights = dLossdOutputWeights.T
                
                # dLossdOutputBias [dLdB] = dLdZ [stored during forward pass] @ dZdB [1]
                dLossdOutputBias = np.sum(layerLocalError, axis=0)

                # dLossdPreviousHiddenLayer [dLdH] = dZdH [layerWeights] @ dLdZ [stored during forward pass]
                dLossdPrevLayerHiddenState = currLayer.layerWeights @ dLossdZ.T
                
                # updating localError to pass back
                layerLocalError = dLossdPrevLayerHiddenState.T

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
                dLossdTimeWeights = prevTimeStepHiddenState.T @ dLossdZ
                # dLossdTimeWeights = dLossdTimeWeights.T
                
                # dLossdLayerWeights [dLdWl] = dZdWl [prevLayerHiddenState] @ dLdZ [calculated above]
                prevLayerHiddenState = currLayer.prevLayerOutputMemory[timeStep]
                dLossdLayerWeights = prevLayerHiddenState.T @ dLossdZ
                # dLossdLayerWeights = dLossdLayerWeights.T
                
                # dLossdOutputBias [dLdB] = dLdZ [calculated above] @ dZdB = [1]
                dLossdOutputBias = np.sum(dLossdZ, axis=0)

                # dLossdPreviousHiddenLayer [dLdH] = dZdH [layerWeights] @ dLdZ [calculated above]
                layerLocalError = currLayer.layerWeights @ dLossdZ.T
                layerLocalError = layerLocalError.T
                
                # dLossdPreviousTimeStep [dLdH] = dZdH [timeWeights] @ dLdZ [calculated above]
                currLayer.thisLayerTimeLocalError = (currLayer.timeWeights @ dLossdZ.T).T
                
                # adding gradients to list for weight updates
                currLayer.layerWeightUpdates += dLossdLayerWeights
                currLayer.timeWeightUpdates += dLossdTimeWeights
                currLayer.biasUpdates += dLossdOutputBias

    # backward pass through entire text
    def backwardPass(self):
        numSteps = self.currentText.shape[1]
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
            currLayer.update(numSteps-1)

    
    def resetGrads(self, train=True):
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
            # currLayer.thisLayerMostRecentOutput = np.zeros(shape=currLayer.thisLayerMostRecentOutput.shape).astype(np.float32)
            if train:
                currLayer.thisLayerMostRecentOutput = np.zeros_like(currLayer.thisLayerMostRecentOutput).astype(np.float32) # hidden state at time 0
            else:
                currLayer.thisLayerMostRecentOutput = np.zeros(size=(currLayer.outputShape))
            # for backward pass calculation
            currLayer.prevLayerOutputMemory = [] # memory of hidden states from the previous layer in  this timestep
            if currLayer.rnn:
                currLayer.prevTimeStepOutputMemory = [] # memory of hidden states from this layer in the previous timestep
            currLayer.thisLayerOutputMemory = [] # memory of the output from this layer

            # for gradient updates
            currLayer.thisLayerTimeLocalError = np.zeros(shape=(currLayer.thisLayerTimeLocalError.shape)).astype(np.float32)
            if currLayer.rnn:
                currLayer.timeWeightUpdates = np.zeros_like(currLayer.timeWeights).astype(np.float32)
            currLayer.layerWeightUpdates = np.zeros_like(currLayer.layerWeights).astype(np.float32)
            currLayer.biasUpdates = np.zeros_like(currLayer.bias).astype(np.float32)

    # training model by repeatedly running forward and backward passes
    def trainModel(self, corpus):
        remainder = len(corpus) % self.batchSize
        if remainder != 0:
            corpus = corpus[:-remainder]
        for count, startIndex in enumerate(range(0, len(corpus), self.batchSize)):
            stopIndex = startIndex + self.batchSize
            batch = corpus[startIndex:stopIndex]
            maxWordCount = max(len(text) for text in batch)
            print(f'Batch #{count+1} - max {maxWordCount} words')
            
            self.trainBatch(batch)
    
    def trainBatch(self, batch):
        # resetting gradients
        self.resetGrads()

        # forward pass
        _ = self.forwardPass(batch)
        modelLoss = self.loss[-1]
        print(f'Loss: {modelLoss}')
        print('********************************************')
        print()

        # backward pass
        self.backwardPass()
    
    # return predicted output for a given input
    def querySequence(self, sequence):
        # getting logits
        logits = self.forwardPass(sequence, train=False)
        
        # determining output word
        outputWordIndex = np.argmax(logits)
        outputWord = list(self.word2ind.keys())[outputWordIndex]

        return outputWord
    

    def generateOutput(self, sequence, numWords):
        self.resetGrads()
        for i in range(numWords):
            nextWord = self.querySequence(sequence)
            sequence.append(nextWord)
        
        return sequence