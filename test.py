from tensorboardX import SummaryWriter
import time
writer = SummaryWriter('runs/exp3')

for i in range(10):
    writer.add_scalar('quadratic', i ** 2, global_step=i)
    writer.add_scalar('exponential', 2 ** i, global_step=i)
    if i ==8:
        break

