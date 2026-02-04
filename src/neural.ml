module Core = struct
  module Utils = Utils
  module Tensor = Tensor
  module Gradients = Gradients
  module Gpu = Gpu
end

module Layers = struct
  module Layer = Layer
  module Linear = Linear
  module Conv2d = Conv2d
  module Activations = Activations
end

module Models = struct
  module Sequential = Sequential
end

module Training = struct
  module Optimizer = Optimizer
  module Errors = Errors
  module Metrics = Metrics
end
