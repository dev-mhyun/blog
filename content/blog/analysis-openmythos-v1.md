---
title: "Analysis of OpenMythos: Long-term Inference Performance of Mythos Preview and Computational Modeling in OpenMythos"
description: "Reading the OpenMythos repository as a computational hypothesis for Mythos Preview's long-horizon capability."
dateString: Apr 2026
draft: false
tags: ["post"]
---

On 2026.04.07, Anthropic announced that its newly released Mythos Preview model had found an absurd number of vulnerabilities in the offensive-security domain. [Mythos Preview](https://red.anthropic.com/2026/mythos-preview/)

In this article I want to analyze [OpenMythos](https://github.com/kyegomez/OpenMythos), a project built on top of the [System Card](https://www-cdn.anthropic.com/8b8380204f74670be75e81c820ca8dda846ab289.pdf) of Mythos Preview — a model that is only offered to specific companies and users.

If you're into security, you've probably already looked at this repo once, made some architectural judgments, and concluded "ah, it's not at the level where you'd actually use it." So let me start by laying out the points this article will focus on.

The core question when reading OpenMythos is: **"With what kind of computational structure can we explain Mythos Preview's long-horizon task capability?"**

OpenMythos is a repository that writes, in PyTorch code, the hypothesis that it is "a model that stably uses more internal computation on long agentic tasks." From this angle, there is quite a lot to learn from OpenMythos's design intent.

Stated simply as a TL;DR, the designed model is this: it increases inference-time compute by repeating the same parameters, re-injects the input so the latent loop doesn't drift, opens a different expert path each iteration with MoE, and gives each position a different amount of computation with ACT.

This article is best read while keeping the following points in mind:

- What kind of computation do the performance patterns in the Mythos Preview System Card demand?
- How did OpenMythos interpret that computation through a Recurrent-Depth Transformer, MoE, MLA, LTIInjection, and ACT?
- Which parts are valid in code, and where do the docs and the implementation diverge?
- What questions does this structure raise from the perspectives of inference-time compute, the hidden latent loop, and safety monitoring?

---

## Performance patterns? Architecture?

The first thing to look at in the Mythos Preview System Card is the *shape* of computation that its performance demands.

The System Card describes Claude Mythos Preview as a frontier model substantially stronger than Anthropic's previous best model, Claude Opus 4.6. In particular it showed strong jumps in software engineering, reasoning, computer use, research assistance, and cybersecurity.

Both I (the author) and you (the reader) are probably most interested in security, so you're likely wondering whether you can actually use it. Since this article is about exploring the design itself as laid out in the repo, I'll cover learning through the architecture and performance tracking on benchmarks in a separate article.

In fact, the reason Mythos Preview is good at offensive work is partly the model architecture, but its performance also comes out well because various foundations (probe classifier, partner vetting, disclosure workflow, Claude Code scaffold, verifier agent, sandbox operations, etc.) back it up. Because OpenMythos deals only with the model itself, my view is that "can you use it right away?" is, of course, no.

### Mythos Preview & OpenMythos

The capability table in the System Card brags: "we pulled these benchmarks!"

| Evaluation | Mythos Preview | Opus 4.6 | Notes |
|---|---:|---:|---|
| SWE-bench Verified | 93.9% | 80.8% | Real-world bug fixing |
| SWE-bench Pro | 77.8% | 53.4% | Harder multi-file tasks |
| SWE-bench Multilingual | 87.3% | 77.8% | Code fixes in 9 languages |
| SWE-bench Multimodal | 59.0% | 27.1% | Includes screenshots and mockups |
| Terminal-Bench 2.0 | 82.0% | 65.4% | Terminal-based tasks |
| GPQA Diamond | 94.5% | 91.3% | High-difficulty science QA |
| USAMO 2026 | 97.6% | 42.3% | Math proofs after the training cutoff |
| GraphWalks BFS 256K–1M | 80.0% | 38.7% | Long-context graph traversal |
| HLE no tools | 56.8% | 40.0% | Humanity's Last Exam |
| HLE with tools | 64.7% | 53.1% | With search, fetch, code execution |

Usually when we see a table like this, we can stop at "oh nice, good model" if the numbers are high. But the thing to look at, more than any single number, is the *shape* of the model's capability.

Mythos Preview is closer to a model that grabs a long task and pushes it through to the end than a model that answers short questions. The same direction shows up in the System Card's qualitative evaluations, which describe the model receiving an engineering objective and carrying out a long cycle of investigation, implementation, testing, and reporting.

Because Cybench is near saturation, Anthropic emphasized more realistic evaluations — CyberGym, a Firefox exploit, and a private cyber range. The capability needed here is not simple knowledge recall, but a structure of:

- read a large codebase,
- pick the spots that look vulnerable,
- judge exploitability,
- develop a crash into a primitive,
- verify it in an execution environment,
- and when it fails, go back to a different hypothesis.

Honestly, I think this cycle is the same loop we run when auditing code ourselves.

This is exactly why OpenMythos brought out structures like recurrent-depth, MoE, and ACT — because the shape of the disclosed performance offers the strongest explainability for the hypothesis of "a model that does more internal computation."

### System Card & OpenMythos

To see why this OpenMythos repo exists, you need to know three facts.

First, training included the public internet, public/private datasets, and synthetic data, and after pretraining there was substantial post-training and fine-tuning.

Second, OpenMythos implements an RDT-style model in `open_mythos/main.py`, `variants.py` provides configurations from 1B to 1T, and `training/3b_fine_web_edu.py` is a FineWeb-Edu streaming pretraining script.

Third, by inference between the two facts, it sets up the hypothesis that Mythos Preview's long-horizon task capability can be explained by recurrent-depth or a latent loop.

---

## OpenMythos Files

OpenMythos has many files, but quite a lot of the modeling ideas are packed into `main.py` alone.

| File | Role |
|---|---|
| `open_mythos/main.py` | OpenMythos core. Implements RDT, MLA/GQA, MoE, LTIInjection, ACT, LoRA |
| `open_mythos/variants.py` | 1B, 3B, 10B, 50B, 100B, 500B, 1T configurations |
| `open_mythos/tokenizer.py` | Hugging Face AutoTokenizer wrapper |
| `open_mythos/moda.py` | Separate MoDA + DeepSeekMoE experimental model |
| `training/3b_fine_web_edu.py` | FineWeb-Edu based 3B pretraining script |
| `test_main.py` | Unit tests for the main modules |
| `docs/open_mythos.md` | API documentation |
| `docs/datasets.md` | Dataset recommendation document |

---

## Core Structure

The most important code in OpenMythos is `OpenMythos.forward()`.

```python
x = self.embed(input_ids)
freqs_cis = (
    self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
)[start_pos : start_pos + T]
mask = self._causal_mask(T, device) if T > 1 else None

for i, layer in enumerate(self.prelude):
    x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

e = x  # encoded input, frozen for injection every loop
x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

for i, layer in enumerate(self.coda):
    x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

return self.head(self.norm(x))
```

OpenMythos actually shows the philosophy of the whole model right in this code.

First it embeds the tokens, and the Prelude layers process the input once; the output up to here is stored as `e`. Then the Recurrent Block takes `x` and `e` together and computes repeatedly several times. Finally the Coda layers post-process and the LM head produces logits.

The key here is this one simple line: `e = x`.

OpenMythos keeps the input representation after the Prelude as a fixed anchor, and the anchor goes back in on every recurrent loop. It's a device to prevent the hidden state from being pulled only toward the representation it created itself while running a long loop.

This design becomes a good hypothesis for explaining Mythos Preview's long agentic-task performance: in long coding tasks the model must keep remembering the initial goal and constraints, but if the hidden state drifts as it loops, the model can grow refined in a direction different from the original request. OpenMythos reduces that risk by injecting the encoded input at every loop.

> Note: this is a solution for *internal* hidden-state drift. Drift in an actual agentic system also arises from tool outputs, file changes, subagent results, stale assumptions, and verifier errors.

---

## Recurrent Block

To understand OpenMythos you have to look at `RecurrentBlock.forward()`.

```python
for t in range(n_loops):
    h_loop = loop_index_embedding(h, t, self.loop_dim)
    combined = self.norm(h_loop + e)
    cache_key = f"recurrent_loop_{t}"
    trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
    trans_out = trans_out + self.lora(trans_out, t)
    h = self.injection(h, e, trans_out)
    p = self.act(h)
    still_running = ~halted
    remainder = (1.0 - cumulative_p).clamp(min=0)
    weight = torch.where(
        cumulative_p + p >= self.cfg.act_threshold,
        remainder,
        p,
    )
    weight = weight * still_running.float()
    h_out = h_out + weight.unsqueeze(-1) * h
```

This code stacks several paper ideas into one loop:

- `loop_index_embedding` tells it which loop it's currently on.
- `h_loop + e` recombines the current hidden state with the input anchor.
- `TransformerBlock` performs the attention and FFN computation.
- The FFN inside the recurrent block is an MoE.
- `LoRAAdapter` compensates for differences in expressiveness per loop.
- `LTIInjection` manages the stability of the recurrent update.
- `ACTHalting` predicts, per position, when to stop.

The goal shows up very clearly in this structure: the aim is to increase inference-time compute without continually growing the parameters.

In a normal transformer, increasing depth means adding layers and growing parameters. OpenMythos increases *effective* depth by running the same block several times. So `n_loops` is not a simple option but a knob used to control the model's reasoning depth.

This is the core way OpenMythos tries to explain Mythos Preview: if Mythos Preview does long tasks well, that capability could be due to larger model size, better post-training, or a stronger agent scaffold. OpenMythos chose — and implemented — the explanation that the inference-time latent loop creates depth.

---

## Latent reasoning

One of the interesting things in OpenMythos is that it put latent reasoning *inside* the model.

In this structure, instead of emitting a chain-of-thought as tokens, the hidden state is updated repeatedly within the same forward pass. That is, the intermediate thought does not remain as text.

Why is this interesting? Because it performs better. Of course, from a safety and stability standpoint it gets a bit harder: reasoning emitted as tokens can at least be read by a monitor, whereas a hidden loop can't be seen by an ordinary transcript monitor. So whether the recurrent update stays stable matters.

OpenMythos addresses this with `LTIInjection`.

```python
def get_A(self) -> torch.Tensor:
    return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

def forward(self, h, e, transformer_out):
    A = self.get_A()
    return A * h + self.B * e + transformer_out
```

The principle is quite simple: `get_A()` always produces a value between 0 and 1, used like a diagonal state matrix. So if you look only at the `A * h` term of the linear recurrence, the spectral radius is less than 1.

This is a stabilization view in the Parcae family: a looped model can have its hidden state blow up as iterations deepen, and binding `A` to a stable region can reduce that risk.

That said, it's a bit of a shame that it doesn't experimentally prove the stability of the *nonlinear* recurrent transformer. `transformer_out` is a nonlinear term that includes attention and MoE, so the implementation doesn't go so far as to prove under what conditions this term is bounded.

---

## Repeating the parameter reduces expressiveness

The advantage of the recurrent-depth structure is parameter efficiency: using the same block several times lets you increase depth without increasing the parameter count.

But, as you can see if you think about it for a moment, the downside is that if every loop runs the same function, it's hard for the first loop and the last loop to play different roles. So initial exploration, intermediate combination, and final verification can get smeared into the same operation form.

OpenMythos handles this problem in two steps.

First, it adds a loop-index embedding.

```python
angles = loop_t * freqs
emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
emb_full[:loop_dim] = emb
return h + emb_full.unsqueeze(0).unsqueeze(0)
```

> In the repo the name is "RoPE-like," but the actual implementation looks closer to a sinusoidal bias than a RoPE rotation.

It adds a loop-index signal to some of the front channels of the hidden state — a device that lets the model know which depth it's currently at even when using the same weights.

Second, it uses depth-wise LoRA.

```python
s = self.scale(torch.tensor(t_idx, device=x.device))
down = self.down(x) * s
return down @ self.B
```

The large weights are shared, but the per-loop low-rank scale is used differently; this approach manages the trade-off between pure weight tying and fully distinct layers.

These two are designs to reduce the expressiveness loss of recurrent-depth, but when `n_loops` exceeds `max_loop_iters`, LoRA reuses the last scale. A completely new phase doesn't appear in loops outside the training range: the loop-index embedding keeps changing while the per-loop difference of the LoRA adapter saturates, so there's a limit to extrapolation. (If you want to demonstrate extrapolation here, you could measure a performance curve by number of loops.)

---

## MoE

In OpenMythos, most of the parameters are concentrated in the recurrent block's MoE.

`TransformerBlock` uses `MoEFFN` only when `use_moe=True`. Prelude and Coda are dense SwiGLU FFNs.

```python
self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
```

The loop creates depth and the MoE creates width. Even when repeating the same recurrent block, if the hidden state changes each loop, the router can select different experts — so different sparse computation paths arise depending on token and loop depth, even within the same parameter set.

Benchmarks show Mythos Preview is strong at coding, browsing, long-context graph traversal, exploits, multimodal SWE, and terminal tasks. To explain that domain breadth you need a wide knowledge-storage path, so OpenMythos seems to express it with MoE.

OpenMythos's MoE implementation is as follows.

```python
logits = self.router(flat)
scores = F.softmax(logits, dim=-1)
_, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
topk_scores = scores.gather(-1, topk_idx)
topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
```

The router adds `router_bias` for expert *selection*, but takes the actual gate weights from the bias-free `scores`. This structure resembles the aux-loss-free routing idea quite a bit.

There's an implementation question here: `router_bias` is registered as a buffer, but no code updates it during training, `training/3b_fine_web_edu.py` has no logic that aggregates expert load to adjust the bias, and the main path has no auxiliary load-balancing loss either — so in effect a load-balanced MoE is not implemented.

This matters in deep-learning terms because if routing collapses in an MoE, the sparse capacity collapses: if only some experts get selected and the rest die, a large parameter count does not translate into actual effective capacity. I mention it here cautiously.

---

## MLA

OpenMythos's default attention is MLA.

MLA is a KV-cache-reducing structure known from DeepSeek-V2: instead of caching full K/V, it stores a latent representation and reconstructs K/V when needed.

```python
kv_raw = self.kv_down(x)
c_kv = kv_raw[..., : self.kv_lora_rank]
k_rope = kv_raw[..., self.kv_lora_rank :]
k_rope = (
    k_rope.unsqueeze(2)
    .expand(B, T, self.n_heads, self.qk_rope_dim)
    .contiguous()
)
k_rope = apply_rope(k_rope, freqs_cis)
```

The cache holds `c_kv` and `k_rope`.

```python
kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}
```

And during attention it reconstructs `k_nope` and `v` from `c_kv`.

One slightly puzzling point, though, is the README's claim of a 10–20× memory reduction. In the current code, `k_rope` is expanded by the number of heads and then cached, so the MLA cache becomes larger than expected. Calculated by variant, in many configurations OpenMythos's MLA cache is actually *larger* than this code's GQA cache.

| Variant | GQA cache/token/layer | MLA cache/token/layer | MLA/GQA |
|---|---:|---:|---:|
| default | 1,024 | 1,536 | 1.50× |
| 1B | 1,024 | 768 | 0.75× |
| 3B | 1,536 | 1,152 | 0.75× |
| 10B | 2,048 | 2,560 | 1.25× |
| 50B | 2,048 | 3,584 | 1.75× |
| 100B | 2,048 | 4,608 | 2.25× |
| 500B | 4,096 | 7,168 | 1.75× |
| 1T | 4,096 | 9,216 | 2.25× |

Looking at the computed table, OpenMythos's MLA *can* reduce the cache compared to full MHA. But going by the current implementation and presets alone, it is not always smaller than GQA. (Cautiously stating my own opinion: there seems to be a bit of a gap between the README's 10–20× reduction and the code.)

---

## RoPE & generation cache

Looking at the Mythos RoPE implementation, it pre-builds complex phasors, treats adjacent pairs of dimensions of the query/key as complex numbers, then applies a per-position rotation — preserving the norm and creating the relative-position property in self-attention.

An important point here is `start_pos`. `OpenMythos.generate()` feeds the entire prompt at the first step, and only the last token at subsequent steps. If the RoPE position keeps coming in as 0 here, the relative position between the cached key and the new query breaks.

```python
if step == 0:
    cur_ids = input_ids
    start_pos = 0
else:
    cur_ids = input_ids[:, -1:]
    start_pos = prompt_len + step - 1

logits = self.forward(
    cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
)
```

(While preparing this post I looked at the GitHub commit log, and it was kind of funny that the log even had a record of fixing a RoPE decode-position bug.)

---

## ACT

ACT is Adaptive Computation Time; it predicts a halting probability for each position.

```python
return torch.sigmoid(self.halt(h)).squeeze(-1)
```

And in `RecurrentBlock`, when the cumulative halting probability exceeds a threshold, it stops that position's contribution to the output.

This design is meaningful in long contexts: every token in a long context has a different processing difficulty. Some tokens only need a grammatical connection, while others must combine far-apart code dependencies or math conditions. Giving every position the same number of loops wastes compute, but ACT enables per-position compute allocation.

---

## Tokenizer & training script

```python
DEFAULT_MODEL_ID = "openai/gpt-oss-20b"

class MythosTokenizer:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
```

The training script is set up to do a FineWeb-Edu streaming dataset, FSDP, AdamW, mixed precision, and checkpointing.

```python
seq_len = 2048
micro_batch = 4
target_tokens = 30_000_000_000
grad_accum = max(1, 256 // (world_size * micro_batch))
...
cfg = mythos_3b()
cfg.vocab_size = vocab_size
cfg.max_seq_len = seq_len
...
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    auto_wrap_policy=wrap_policy,
    device_id=local_rank,
)
```

That said, the information you get during the pretraining process is a bit lacking:

- no validation loop
- no benchmark evaluation
- no MoE expert-utilization logging
- no ACT average-loop-count logging
- no hidden-norm-by-loop logging
- no router load-balancing update
- the streaming dataset's resume position is not restored

In particular, for a model that includes MoE and ACT, without internal metrics it's hard to know the cause of a training failure — from loss alone you can't tell whether it's expert collapse, halting collapse, or a recurrent-norm problem.

---

## Conclusion: OpenMythos's Value

The OpenMythos repo explains the long-horizon task capability shown by the Mythos System Card with a single computational model:

- Depth comes not from a bigger layer stack but from the recurrent loop.
- Width is secured by the MoE expert pool inside the recurrent loop.
- Stability is managed with input reinjection, an LTI-stable update, a loop-index signal, and ACT.

From this perspective, OpenMythos is a repo that quite nicely codes up the question, "Can long-horizon agentic reasoning be explained by the model's internal repeated computation?"

Of course it's not a finished training run, nor a verified offensive model that we can use.

Even so, it's well worth reading, because going forward, capable coding agents and cyber agents will not be explained by parameter count alone.

Through recent news, I hear that many people want to build their own task-forced Local Agents or Local AI.

So I'll close by sharing a personal thought: if — through repos like this — you understand where more inference-time compute is spent, how to stabilize latent reasoning that isn't visible as tokens, how to keep the hidden state from drifting on long tasks, and how far a safety monitor can see into a hidden loop, it could be useful when building a task-forced AI.

(It was a long one — thank you for reading. I'm still studying and growing, so if you spot mistakes or shortcomings in the content, please send feedback to devmhyun@gmail.com and I'll learn from it. Thanks again for reading.)

---

## References

- Anthropic, [Assessing Claude Mythos Preview's cybersecurity capabilities](https://red.anthropic.com/2026/mythos-preview/)
- Anthropic, [Claude Mythos Preview System Card (PDF)](https://www-cdn.anthropic.com/8b8380204f74670be75e81c820ca8dda846ab289.pdf)
- Anthropic, [Model system cards](https://www.anthropic.com/system-cards/)
- Harsh Kohli et al., [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822)
- Hayden Prairie et al., [Parcae: Scaling Laws For Stable Looped Language Models](https://arxiv.org/abs/2604.12946)
- Nikunj Saunshi et al., [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- Damai Dai et al., [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- Joshua Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Alex Graves, [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)
- Jianlin Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Biao Zhang and Rico Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- PyTorch, [FullyShardedDataParallel documentation](https://docs.pytorch.org/docs/stable/fsdp.html)
- Hugging Face, [Transformers AutoTokenizer documentation](https://huggingface.co/docs/transformers/model_doc/auto)
- Hugging Face, [Datasets streaming documentation](https://huggingface.co/docs/datasets/main/en/stream)
