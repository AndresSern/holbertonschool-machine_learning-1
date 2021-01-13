#!/usr/bin/env python3
def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    a0 =alpha
    a = global_step / decay_step
    a = int(a)
  
    
    alpha = a0 *(1/ (1+(decay_rate * a)))
        
    return alpha