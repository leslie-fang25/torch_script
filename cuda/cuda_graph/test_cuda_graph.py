# rm -rf /tmp/torchinductor_leslie/* && rm -rf torch_compile_debug/* && clear && TORCH_LOGS="+schedule,+inductor,+output_code" TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_FREEZING=1 numactl -C 56-56 --membind=1 python arrange.py

# Test Commit: 7a694f66835ab18512a723c1761f2945c831416f
# ref time is: 28.353589296340942
# inductor time is: 30.05754566192627

# Ref Inductor Commit: 8619fe6214cd8f31345ae73c5b90024a0233dc40
# ref time is: 29.15333342552185
# inductor time is: 35.822702407836914

import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch._inductor.config

local_seed= 2024

torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

dtype = torch.bfloat16
autocast = True if dtype == dtype else False

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return F.layer_norm(input, (1039,))

if __name__ == "__main__":

    with torch.no_grad():
        m = M().eval().to("cuda")
        
        static_input = torch.randn(1, 384, 1039, device="cuda").to(dtype)
        static_output = torch.randn(1, 384, 1039, device="cuda").to(dtype)

        warmup_steps = 3
        steps = 5

        with torch.no_grad():
            
            # Warmup before capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup_steps):
                    m(static_input)
            torch.cuda.current_stream().wait_stream(s)

            # capture graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output = m(static_input)
                static_output.copy_(output)
            
            # 回放
            ref_start = time.time()
            for _ in range(steps):
                new_input = torch.randn(1, 384, 1039, device="cuda").to(dtype)
                # 每次的输入都需要copy 到 static_input 里面
                static_input.copy_(new_input)
                g.replay()
                # 执行结果放在 static_output 里面
                print("static_output is: {}".format(static_output), flush=True)
            ref_end = time.time()
    

