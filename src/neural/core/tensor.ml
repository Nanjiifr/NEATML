type cpu_tensor = float array array
type gpu_tensor = Gpu.tensor

type t = 
  | CPU of cpu_tensor
  | GPU of gpu_tensor
