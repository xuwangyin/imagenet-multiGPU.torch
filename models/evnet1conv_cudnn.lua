function createModel(nGPU)
   require 'torch'   -- torch
   require 'image'   -- for image transforms
   require 'nn'      -- provides all sorts of trainable modules/layers

   -- 10-class problem
   local nfeats = 1
   local width = 32
   local height = 32
   local ninputs = nfeats*width*height
   local noutputs = 2

   -- number of hidden units (for MLP only):
   local nhiddens = ninputs / 2

   local nstates = {32,32,64}
   local filtsize = 5
   local poolsize = 2
   local normkernel = image.gaussian1D(7)
   local model

   if true then
      -- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      model:add(nn.SpatialConvolutionMM(nfeats, 32, filtsize, filtsize))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      model:add(nn.View(32 * 14 *14))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(32 * 14 * 14, nstates[3]))
      model:add(nn.ReLU())
      model:add(nn.Linear(nstates[3], noutputs))
      model:add(nn.LogSoftMax())
   else
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))
   end
   return model
end
