# Mixed Precision Training
## Two Kinds of `float`
FP16 and FP 32 are all floats. FP16 takes 2 bytes while FP32 takes 4 bytes.
Compared to FP16, FP32 has greater range and is preciser. But it takes more memory and time during computation.
It's common to meet CUDA `Out Of Memory` exception.
![[截屏2025-11-07 11.34.52.png]]
But when we are using FP16 throughout the training, some small gradient values will **underflow(become zero)**. We should use both FP16 and FP32.
## Mixing FP32 and FP16
![[截屏2025-11-07 11.39.52.png]]
For origin model parameters, we all use FP32 to store them.
When passing to the training process(Forward, Backward and Compute Gradient), we cut them to FP16.
We can scale the final loss by a constant factor to further reduce the possibility of underflow.
Before doing gradient operations(gradient clipping, updating the original model parameters), we unscale the gradient(divide by scale factor).
## Implement in PyTorch
We first initialize a `GradScaler`.
We can assign the FWD and Loss Calculation process explicitly
```Python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
```
The `step` function of scaler will automatically `unscale` the gradient.
But if we need to do gradient clipping, we should unscale it explicitly.
```Python
# Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)
```
## BFloat16
![[截屏2025-11-07 12.01.48.png]]
BFloat16 is a special float type.
It can express much smaller(closer to 0) or larger values than FP16, for it has 8 Exponents.
But it can't be much precise.
Involving BFloat16(replace FP16 with BFloat16), we then need no gradient scaling.
# Multi-GPU Training
## Distributed Data Parallel(DDP)
When we have a few GPUs, to implement DDP, we store a copy of the whole model parameters in each GPU.
![[截屏2025-11-07 21.26.04.png]]
When do batched-training, we equally divide the batch into parts.(For example, there are 64 data point in one batch, and we have 8 GPUs, so we assign 8 data point to each GPU)
After calculating the loss and the gradient of each GPU's batches, all GPUs communicate with each other and average each one's gradient(this process is called "AllReduce"), and do the parameter updating.
Then load the next batch and continue training.
However, each GPU holds a full copy of model parameters, optimizer state and gradients. It's a waste of memory.
## Fully Sharded Data Parallel(FSDP)
FSDP also needs to divide the batch into several parts.
GPUs need to communicate with each other, sending values needed by other GPUs.
![[截屏2025-11-09 21.17.16.png]]
In stage 1($P_{os}$) and stage 2($P_{os+g}$), we distribute the optimizer states(stage1 and stage 2) and the gradient(stage 2), each GPU only holding a shard of the full datas.
These process won't cause a time cost.
To stage 3($P_{os+g+p}$), we even distribute the parameters. This will save a lot of memory, but need to cost more time.
The full process of FSDP goes as follows.
1. Divide model parameters into FSDP units, and shard each unit across multiple GPUs.
![[截屏2025-11-09 21.30.41.png]]
2. Run forward pass.
	- For each layer, perform an all-gather so each GPU gets what it needs.
	- Run forward pass.
	- Discard used parts of each GPU.
![[截屏2025-11-09 21.33.55.png]]
3. Run backward pass.
	- For each layer, perform an all-gather so each GPU gets what it needs.
	- Each GPU computes gradient for its data chunk.
	- Do a reduce-scatter(send full gradient piece to the right GPU).
	- Each GPU updates its own shard using the full gradient received earlier.
![[截屏2025-11-09 21.38.39.png]]
# Parameter-efficient Fine-tuning(PEFT)
To finetune a large model, we need huge memory. So we shall come up with some methods.
First we can only choose and finetune only a small part of the model.
We do forward propagation with full parameters and only do gradient step to some parameters and fix other parameters.
![[截屏2025-11-09 21.39.53.png]]
## Low-rank-parameterized Update(LoRA)
When doing weight update, the updating matrix usually is a low rank matrix.
For a weight matrix $W_0$, we do
$$
W'=W_0+\Delta W
$$
in which, Three matrices are $d\times k$.
Notice that $W_0$ is weight matrix after pretraining. $\Delta W$ is the overall difference between the pretrained weight and the finetuned weight.
And here, $\Delta W$ is usually a low-rank matrix.
So, we can view it as
$$
\Delta W=BA
$$
in which, $B\in R^{d\times r}$ and $A\in R^{r\times k}$.
$r$ is far small than $\min(d,k)$, causing the number of parameters to be update greatly reduced.
While finetuning, we fix $W_0$ and only modify $B$ and $A$.
After the training process, we do update with hyperparameter $\alpha$
$$
W_+=W_0+\alpha BA
$$
in which, $\alpha$ is a scalar that do a trade-off between original knowledge learned from pretraining and new knowledge learned from finetuning.
# Strategy Selection in Efficient Training
We shall use different efficient training methods, for our GPU power and memory are always finite.
![[截屏2025-11-09 22.30.24.png]]