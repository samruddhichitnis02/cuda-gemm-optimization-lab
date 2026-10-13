import os
import csv
import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_matmul_bias_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add bias
    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    acc = acc + bias[None, :]

    # GELU (tanh form): x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3))
    z = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)
    acc = acc * tl.sigmoid(2.0 * z)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def triton_fused(a, b, bias):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % 128 == 0 and N % 128 == 0 and K % 32 == 0, "sizes must be multiples of 128"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    fused_matmul_bias_gelu_kernel[grid](
        a, b, bias, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def eager_fn(a, b, bias):
    return F.gelu(a @ b + bias, approximate="tanh")


def time_ms(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch", torch.__version__, "| triton", triton.__version__)

    compiled_fn = torch.compile(eager_fn)

    sizes = [1024, 2048, 4096]
    rows = []
    header = f"{'N':<6}{'impl':<16}{'median_ms':>10}{'TFLOPS':>10}{'max_err':>12}  status"
    print()
    print(header)

    for n in sizes:
        M = N = K = n
        scale = K ** -0.25
        a = (torch.randn(M, K, device="cuda") * scale).half()
        b = (torch.randn(K, N, device="cuda") * scale).half()
        bias = torch.randn(N, device="cuda").half()

        ref32 = F.gelu(a.float() @ b.float() + bias.float(), approximate="tanh")
        flops = 2.0 * M * N * K

        impls = [
            ("eager", lambda: eager_fn(a, b, bias)),
            ("torch_compile", lambda: compiled_fn(a, b, bias)),
            ("triton_fused", lambda: triton_fused(a, b, bias)),
        ]

        for name, fn in impls:
            try:
                out = fn()
                torch.cuda.synchronize()
                err = (out.float() - ref32).abs().max().item()
                status = "PASS" if err < 3e-2 else "FAIL"
                ms = time_ms(fn)
                tflops = flops / (ms * 1e-3) / 1e12
                extra = ""
                if name == "triton_fused":
                    extra = f"  config={getattr(fused_matmul_bias_gelu_kernel, 'best_config', None)}"
                print(f"{n:<6}{name:<16}{ms:>10.4f}{tflops:>10.2f}{err:>12.2e}  {status}{extra}")
                rows.append([n, name, round(ms, 4), round(tflops, 2), f"{err:.2e}", status])
            except Exception as e:
                print(f"{n:<6}{name:<16} ERROR: {type(e).__name__}: {str(e)[:150]}")
                rows.append([n, name, "", "", "", "ERROR"])
        print()

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "fused_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "impl", "median_ms", "tflops", "max_err", "status"])
        w.writerows(rows)
    print("Saved results/fused_results.csv")


if __name__ == "__main__":
    main()