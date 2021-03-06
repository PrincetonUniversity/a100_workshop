# What GPU resources does Princeton Research Computing have?

See the [Hardware Resources](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#hardware) on the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for a complete list.

# Adroit

![Adroit Diagram](../01_a100_overview/images/adroit_2021-10-18.png)

```
$ ssh <YourNetID>@adriot.princeton.edu
$ shownodes  # or snodes
NODELIST      PART   STATE      FREE/TOTAL CPUs  CPU_LOAD  FREE/TOTAL MEMORY  FREE/TOTAL GPUs   FEATURES
adroit-01     class  idle                 28/28      0.00    120536/128000Mb                   broadwell
adroit-02     class  idle                 28/28      0.00    120551/128000Mb                   broadwell
adroit-03     class  idle                 28/28      0.00    120597/128000Mb                   broadwell
adroit-04     class  idle                 28/28      0.54    120595/128000Mb                   broadwell
adroit-05     class  idle                 28/28      0.00    120620/128000Mb                   broadwell
adroit-06     class  idle                 28/28      0.00    120606/128000Mb                   broadwell
adroit-07     class  idle                 28/28      0.00    120608/128000Mb                   broadwell
adroit-08     all    mixed                 6/32     20.25    195921/384000Mb                     skylake
adroit-09     all    mixed                 7/32     21.42    210347/384000Mb                     skylake
adroit-10     all    drained               0/32      0.00    383312/384000Mb                     skylake
adroit-11     all    drained               0/32      0.37    383437/384000Mb                     skylake
adroit-12     all    idle                 32/32      0.00    383504/384000Mb                     skylake
adroit-13     all    mixed                25/32      3.31    271857/384000Mb                     skylake
adroit-14     all    mixed                31/32      1.00    382839/384000Mb                     skylake
adroit-15     all    mixed                28/32      4.42    379556/384000Mb                     skylake
adroit-16     all    mixed                 4/32      9.16    177025/384000Mb                     skylake
adroit-h11g1  gpu    mixed                39/40      1.07    761845/770000Mb   3/4 tesla_v100       v100
adroit-h11g2  gpu    idle                 48/48      0.00  1014561/1000000Mb  4/4 nvidia_a100       a100
adroit-h11n1  class  mixed              109/128      0.01    238074/256000Mb                    amd,rome
```

## Hands-on Exercise

The schematic diagram below is for adroit-h11g2 which is composed of two CPUs and four GPUs:

![adroit-h11g2](../01_a100_overview/images/adroit_a100_gpu_nodes.png)

Learn about NVIDIA GPU nodes on Adroit by running the following commands:

```bash
$ snodes
$ ssh adroit-h11g1
(adroit-h11g1) $ nvidia-smi -a | less  # press q to quit
(adroit-h11g1) $ nvidia-smi topo -m
(adroit-h11g1) $ lscpu
(adroit-h11g1) $ exit
$ ssh adroit-h11g2
(adroit-h11g2) $ nvidia-smi -a | less  # press q to quit
(adroit-h11g2) $ nvidia-smi topo -m
(adroit-h11g2) $ lscpu
(adroit-h11g2) $ exit
```

Which NVIDIA A100 GPU is available on adroit-h11g2 (40GB PCIe, 80GB PCIe, 40GB SXM or 80GB SXM)?

## Topology

The follow command provides insight into how the GPUs are connected or not connected:

```
[aturing@adroit-h11g2 ~]$ nvidia-smi topo -m
        GPU0	GPU1	GPU2	GPU3	mlx5_0	CPU Affinity	NUMA Affinity
GPU0	 X 	NODE	SYS	SYS	NODE	0,2,4,6,8,10	0
GPU1	NODE	 X 	SYS	SYS	NODE	0,2,4,6,8,10	0
GPU2	SYS	SYS	 X 	NODE	SYS	1,3,5,7,9,11	1
GPU3	SYS	SYS	NODE	 X 	SYS	1,3,5,7,9,11	1
mlx5_0	NODE	NODE	SYS	SYS	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
 
`mlx5_0` is the Mellanox network adapter. SYS indicates P2P and GPUDirect RDMA transactions cannot follow that path.

For comparison see the [topology for Traverse](https://researchcomputing.princeton.edu/systems/traverse#smt) which gives:

```
[aturing@traverse-k02g8 ~]$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	mlx5_0	mlx5_1	mlx5_2	mlx5_3	CPU Affinity	NUMA Affinity
GPU0	 X 	NV3	SYS	SYS	NODE	NODE	SYS	SYS	0-63	0
GPU1	NV3	 X 	SYS	SYS	NODE	NODE	SYS	SYS	0-63	0
GPU2	SYS	SYS	 X 	NV3	SYS	SYS	NODE	NODE	64-127	8
GPU3	SYS	SYS	NV3	 X 	SYS	SYS	NODE	NODE	64-127	8
mlx5_0	NODE	NODE	SYS	SYS	 X 	PIX	SYS	SYS		
mlx5_1	NODE	NODE	SYS	SYS	PIX	 X 	SYS	SYS		
mlx5_2	SYS	SYS	NODE	NODE	SYS	SYS	 X 	PIX		
mlx5_3	SYS	SYS	NODE	NODE	SYS	SYS	PIX	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

Here is the topology for the new CryoEM nodes which may be used for TigerGPU:

```
[aturing@della-xxxxx ~]$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	mlx5_0	CPU Affinity	NUMA Affinity
GPU0	 X 	NV4	NV4	NV4	NODE	0-23	0
GPU1	NV4	 X 	NV4	NV4	NODE	0-23	0
GPU2	NV4	NV4	 X 	NV4	SYS	24-47	1
GPU3	NV4	NV4	NV4	 X 	SYS	24-47	1
mlx5_0	NODE	NODE	SYS	SYS	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

What is the GPU topology of the [della-gpu nodes](https://researchcomputing.princeton.edu/systems/della#gpus)?


### NUMA domains on the CPU

Each Adroit GPU node has two CPUs as revealed by lscpu. The memory of the two CPUs is made to behave like one large memory pool using NUMA (non-uniform memory access). Use the command below to see which CPU-cores belong to which NUMA domain:

```
$ ssh adroit-h11g2
$ numactl -H
available: 2 nodes (0-1)
node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46
node 0 size: 515192 MB
node 0 free: 506918 MB
node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47
node 1 size: 516088 MB
node 1 free: 507630 MB
node distances:
node   0   1 
  0:  10  20 
  1:  20  10
```

We see that there are two NUMA domains. The matrix at the bottom of the output shows the penalty for accessing one domain from another.

### Theoretical Performance of adroit-h11g2

Peak FP64 performance of the 4 GPUs is:

```
4 x 9.7 TFLOPS = 38.8 TFLOPS
```

Peak performance of the 2 CPUs:

```
2 CPUs x 24 cores/CPU x 2.8e9 cycles/second x 2 VPUs/core x 2 FMA/cycle/VPU x 8 FP64 operations/FMA = 4.3 TFLOPS
```

VPU is vector processing unit, FMA is fused multiply-add, 8 OPS/VPU is the width of a VPU for double precision numbers. This calculation shows that the GPUs are roughly four times more powerful than the 2 CPUs.

## Bandwidth Test

GPU memory reads are extremely fast and need to be since there are thousands of execution units that must be fed. But what about the bandwidth of transfer between the CPU and the GPU?

Take a look at the [Della-GPU](https://researchcomputing.princeton.edu/systems/della#gpus) schematic to see the reported bandwidth.

![della_gpu](https://researchcomputing.princeton.edu/sites/g/files/toruqf311/files/styles/freeform_2880w/public/2021-05/della_gpu_node_v4_1600.png?itok=sWnS80yR)

Run the commands below to measure various memory bandwidths on the Adroit A100 node:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/gpfs/$USER
$ /usr/local/cuda-11.4/bin/cuda-install-samples-11.4.sh .
$ cd NVIDIA_CUDA-11.4_Samples/1_Utilities/bandwidthTest
$ module load cudatoolkit/11.4
$ make
$ sbatch job.slurm  # create this file from below
```

Below is job.slurm:

```bash
#!/bin/bash
#SBATCH --job-name=bandwidth     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (default is 4 GB per CPU-core)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        $ v100 or a100

./bandwidthTest --device=all --memory=pinned --mode=range --start=10000000 --end=100000000 --increment=10000000 --htod --dtoh --dtod
```

Below are the results for multiple machines:

```
============= TIGERGPU ============= 
Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 12.1
  10000000 12.4
 100000000 12.4
1000000000 12.4

Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 12.8
  10000000 13.1
 100000000 13.1
1000000000 13.1

Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 551.5
  10000000 460.2
 100000000 516.3
1000000000 522.6


============= DELLA-GPU =============
Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 18.0
  10000000 19.4
 100000000 24.1
1000000000 25.8

Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 15.8
  10000000 25.4
 100000000 26.2
1000000000 25.7

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 555.5
  10000000 1602.7
 100000000 1268.0
1000000000 1251.4


============= TRAVERSE =============
Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 54.1
  10000000 65.2
 100000000 72.4
1000000000 72.7

Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 55.1
  10000000 64.0
 100000000 71.0
1000000000 72.7

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 316.6
  10000000 385.7
 100000000 754.0
1000000000 778.0


============= DELLA-i16g1 =============
$ env CUDA_VISIBLE_DEVICES=0 ./bandwidthTest --device=all --memory=pinned --mode=range --start=100000 --end=1000000 --increment=100000 --htod --dtoh --dtod

OR

$ env CUDA_VISIBLE_DEVICES=1 ./bandwidthTest --device=all --memory=pinned --mode=range --start=100000 --end=1000000 --increment=100000 --htod --dtoh --dtod
=======================================
Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000  7.3
  10000000 24.5
 100000000 24.6
1000000000 24.6

Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 13.7
  10000000 13.9
 100000000 13.8
1000000000 13.9

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000  504.4
  10000000 1576.1
 100000000 1535.9
1000000000 1608.6

============= DELLA-i16g1 =============
$ env CUDA_VISIBLE_DEVICES=2 ./bandwidthTest --device=all --memory=pinned --mode=range --start=100000 --end=1000000 --increment=100000 --htod --dtoh --dtod

OR

$ env CUDA_VISIBLE_DEVICES=3 ./bandwidthTest --device=all --memory=pinned --mode=range --start=100000 --end=1000000 --increment=100000 --htod --dtoh --dtod
=======================================
Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 23.9
  10000000 24.5
 100000000 24.6
1000000000 24.6

Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000 25.3
  10000000 26.0
 100000000 26.3
1000000000 26.3

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes) Bandwidth(GB/s)
   1000000  504.4
  10000000 1576.1
 100000000 1535.9
1000000000 1608.6
```

Why is it important to know these values? When deciding how to solve a problem and whether it is feasible. If you are not seeing the performance you expect then run this test to see what is the peak measured bandwidth.
