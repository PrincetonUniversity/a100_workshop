# Leveraging the NVIDIA A100 GPU for AI and HPC

The NVIDIA A100 GPU will soon be found across the majority of the Research Computing clusters. This powerful accelerator offers a theoretical performance of 9.7 TFLOPS in double precision and 19.5 in single. Specialized hardware units on the GPUs called Tensor Cores allow for even faster speeds. In order to take full advantage of the A100, most applications require users to modify their input scripts.

This workshop provides an overview of the features of the A100 GPU along with specific use cases for deep learning (PyTorch and TensorFlow) and HPC. Tools for performance profiling and for measuring data transfer rates will be presented.

## Local Self-Study (Non-Slurm)

If you already have a local machine with an NVIDIA A100, use [LOCAL_SELF_STUDY.md](LOCAL_SELF_STUDY.md). It adapts the workshop into a workstation-friendly path with local wrapper scripts under `scripts/local/`.

## Princeton / Slurm Workshop Flow

If you are following the original cluster-based workshop flow, start with [setup.md](setup.md) and then continue through the numbered directories in order.

<!--
## Workshop Survey

Toward the end of the workshop please complete [this survey](https://forms.gle/rrBLgZYPjyLHYxFR7).
-->

## Getting Help

If you encounter any difficulties with the material in this guide then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.

## Authorship

This guide was created by Xuefei Zhang, Jonathan Halverson, and members of Princeton Research Computing.
