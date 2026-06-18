---
title: "OpenMythos 분석: Mythos Preview의 장기 추론 성능과 OpenMythos의 계산 모델링"
description: "Mythos Preview의 장기 작업 능력을 OpenMythos 레포의 계산 구조로 해석해 본다."
dateString: Apr 2026
draft: false
tags: ["post"]
translationKey: "openmythos-analysis"
lang: "KR"
---

2026.04.07 Anthropic에서 공개한 Mythos Preview 모델은 Offensive 분야에서 말도 안 되게 많은 취약점을 찾아냈다고 발표했습니다. [Mythos Preview](https://red.anthropic.com/2026/mythos-preview/)

이 글에서는 특정 기업 및 사용자에게만 제공해 사용하고 있는 Mythos Preview의 [System Card](https://www-cdn.anthropic.com/8b8380204f74670be75e81c820ca8dda846ab289.pdf)를 기반으로 개발된 [OpenMythos](https://github.com/kyegomez/OpenMythos)를 분석해보고자 합니다.

보안에 관심 있으신 분이라면 해당 레포를 이미 한 번 보고 아키텍처적인 설계를 했고, "아, 사용할 정도는 아니다" 라는 선에서 분석을 했을 것입니다. 그래서 이 글에서 볼 포인트들을 미리 정리하면서 글을 시작하겠습니다.

OpenMythos를 읽는 핵심 질문은 **"Mythos Preview의 장기 작업 능력을 어떤 계산 구조로 설명할 수 있는가"** 입니다.

OpenMythos는 "장기 agentic task에서 더 많은 내부 계산을 안정적으로 쓰는 모델"이라는 가설을 PyTorch 코드로 쓴 레포입니다. 이 관점에서 보면 OpenMythos의 설계 의도에서 배울 점이 꽤 많습니다.

단순하게 TL;DR로 설명하면 이렇습니다. 같은 parameter를 반복해 inference-time compute를 늘리고, latent loop가 drift하지 않도록 입력을 다시 주입하며, MoE로 반복마다 다른 expert path를 열고, ACT로 position별 계산량을 다르게 주는 방식으로 설계했다.

본 글은 구체적으로 아래 내용을 기반으로 포인트를 짚으면서 보면 좋습니다.

- Mythos Preview System Card의 성능 패턴은 어떤 계산을 요구하는가?
- OpenMythos는 그 계산을 Recurrent-Depth Transformer, MoE, MLA, LTIInjection, ACT로 어떻게 해석했는가?
- 코드상 유효한 부분은 무엇이고, 문서와 구현이 어긋나는 부분은 무엇인가?
- 이 구조가 inference-time compute, hidden latent loop, safety monitoring 관점에서 던지는 질문은 무엇인가?

---

## Performance patterns? Architecture?

Mythos Preview System Card에서 먼저 봐야 할 것은 성능이 요구하는 계산의 형태입니다.

System Card는 Claude Mythos Preview를 Anthropic의 이전 최고 모델인 Claude Opus 4.6보다 크게 강해진 frontier model로 설명합니다. 특히 소프트웨어 엔지니어링, 추론, 컴퓨터 사용, 연구 보조, 사이버 보안 영역에서 강한 점프를 보였다고 합니다.

글을 쓰는 필자도, 글을 읽으시는 독자분들도 가장 관심 있는 대목은 보안일 거라고 생각합니다. 그래서 "이거 사용할 수 있나"에 대한 의문이 들 것 같습니다. 이 글은 레포에서 설계한 설계 자체에 대해 알아보는 것이기 때문에, 아키텍처적인 부분을 통한 학습과 벤치에 대한 성능 추적 등은 다른 글에서 다뤄보도록 하겠습니다.

사실 Mythos Preview가 Offensive를 잘하는 이유는 모델의 아키텍처단도 있지만, 여러 가지(probe classifier, partner vetting, disclosure workflow, Claude Code scaffold, verifier agent, sandbox 운영 체계 등) 기저가 뒷받침되어 성능이 좋게 나옵니다. OpenMythos는 모델 본체에 대한 내용만 다루기 때문에 "바로 실사용할 수 있는가?"는 당연히 아니라고 필자는 생각합니다.

### Mythos Preview & OpenMythos

System Card의 capability table은 이런 벤치를 뽑았다! 라고 자랑합니다.

| 평가 | Mythos Preview | Opus 4.6 | 해석 |
|---|---:|---:|---|
| SWE-bench Verified | 93.9% | 80.8% | 실전 버그 수정 |
| SWE-bench Pro | 77.8% | 53.4% | 더 어려운 multi-file task |
| SWE-bench Multilingual | 87.3% | 77.8% | 9개 언어 코드 수정 |
| SWE-bench Multimodal | 59.0% | 27.1% | 스크린샷과 mockup 포함 |
| Terminal-Bench 2.0 | 82.0% | 65.4% | 터미널 기반 작업 |
| GPQA Diamond | 94.5% | 91.3% | 고난도 과학 QA |
| USAMO 2026 | 97.6% | 42.3% | 훈련 컷오프 이후 수학 증명 |
| GraphWalks BFS 256K–1M | 80.0% | 38.7% | 긴 context의 그래프 탐색 |
| HLE no tools | 56.8% | 40.0% | Humanity's Last Exam |
| HLE with tools | 64.7% | 53.1% | 검색, fetch, code execution 포함 |

보통 우리가 이런 표를 보면 대충 높으면 "오 굿굿 좋은 모델"로 끝날 수 있는데, 어떤 수치보다 봐야 할 점은 모델 수행 능력의 모양입니다.

Mythos Preview는 짧은 질문에 답하는 모델이라기보다 긴 작업을 붙잡고 끝까지 밀어붙이는 모델에 가깝습니다. System Card의 정성 평가에서도 같은 방향이 보이는데, 모델은 engineering objective를 받고 조사, 구현, 테스트, 보고까지 긴 사이클을 수행한다고 설명하고 있습니다.

Cybench는 포화에 가깝기 때문에 Anthropic은 더 현실적인 평가로 CyberGym, Firefox exploit, private cyber range를 강조했는데, 여기서 필요한 능력은 단순 지식 회상이 아니라

- 큰 코드베이스를 읽는다.
- 취약해 보이는 지점을 고른다.
- exploitability를 판단한다.
- crash를 primitive로 발전시킨다.
- 실행 환경에서 검증한다.
- 실패하면 다른 가설로 돌아간다.

라는 구조를 가져야 한다는 것입니다. 사실 해당 사이클은 우리가 코드를 오딧할 때도 똑같이 수행하는 루프라고 생각합니다.

OpenMythos가 recurrent-depth, MoE, ACT 같은 구조를 들고 나온 이유가 여기에 있는데, 공개된 성능의 모양이 "더 많은 내부 계산을 하는 모델"이라는 가설에 대해 가장 큰 설명 가능함을 보여주기 때문입니다.

### System Card & OpenMythos

OpenMythos 레포의 존재 이유를 보려면 3가지 사실을 알아야 합니다.

첫째, 학습에는 public internet, public/private dataset, synthetic data가 포함되었고, pretraining 이후 substantial post-training과 fine-tuning이 있었다.

둘째, OpenMythos는 `open_mythos/main.py`에 RDT 형태의 모델을 구현하고, `variants.py`는 1B부터 1T까지의 설정을 제공하며, `training/3b_fine_web_edu.py`는 FineWeb-Edu streaming pretraining script를 짰다.

셋째, 두 사실 사이의 추론을 통해 Mythos Preview의 장기 작업 능력은 recurrent-depth나 latent loop로 설명될 수 있다는 가설을 세웠다.

---

## OpenMythos Files

OpenMythos는 파일 수가 많은데, `main.py` 하나에 꽤 많은 모델링 아이디어가 몰려 있습니다.

| 파일 | 역할 |
|---|---|
| `open_mythos/main.py` | OpenMythos 본체. RDT, MLA/GQA, MoE, LTIInjection, ACT, LoRA 구현 |
| `open_mythos/variants.py` | 1B, 3B, 10B, 50B, 100B, 500B, 1T 설정 |
| `open_mythos/tokenizer.py` | Hugging Face AutoTokenizer wrapper |
| `open_mythos/moda.py` | 별도 MoDA + DeepSeekMoE 실험 모델 |
| `training/3b_fine_web_edu.py` | FineWeb-Edu 기반 3B pretraining script |
| `test_main.py` | 주요 모듈 단위 테스트 |
| `docs/open_mythos.md` | API 문서 |
| `docs/datasets.md` | 데이터셋 추천 문서 |

---

## Core Structure

OpenMythos의 가장 중요한 코드는 `OpenMythos.forward()`입니다.

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

사실 OpenMythos는 이 코드에서 모델 전체의 철학을 그대로 보여주고 있습니다.

먼저 token을 embedding하고, Prelude layer들이 입력을 한 번 처리합니다. 여기까지의 출력이 `e`로 저장됩니다. 이후 Recurrent Block은 `x`와 `e`를 같이 받아 여러 번 반복 계산합니다. 마지막으로 Coda layer들이 후처리하고 LM head가 logits를 냅니다.

여기서 핵심은 `e = x` 라는 이 간단한 코드입니다.

OpenMythos는 Prelude를 지난 입력 표현을 고정 anchor로 보관하고, anchor는 recurrent loop마다 다시 들어갑니다. 긴 loop를 돌리면서 hidden state가 자기 자신이 만든 표현에만 끌려가는 것을 막기 위한 장치입니다.

이 설계가 Mythos Preview의 long agentic task 성능을 설명하는 데 좋은 가설이 되는데, 긴 코드 작업에서 모델은 계속 초기 목표와 제약을 기억해야 하는데 hidden state가 loop를 돌며 drift하면 모델은 처음 요청과 다른 방향으로 정교해질 수 있습니다. OpenMythos는 encoded input을 매 loop에 주입해 그 위험을 줄이는 것입니다.

> 다만 이건 내부 hidden state drift에 대한 해결책으로, 실제 agentic system의 drift는 tool output, 파일 변경, subagent 결과, 오래된 가정, verifier 오류에서도 발생합니다.

---

## Recurrent Block

OpenMythos를 이해하려면 `RecurrentBlock.forward()`를 봐야 합니다.

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

이 코드는 여러 논문 아이디어를 한 loop 안에 쌓습니다.

- `loop_index_embedding`은 지금 몇 번째 loop인지 알려줍니다.
- `h_loop + e`는 현재 hidden state와 입력 anchor를 다시 결합합니다.
- `TransformerBlock`은 attention과 FFN 계산을 수행합니다.
- recurrent block 안의 FFN은 MoE입니다.
- `LoRAAdapter`는 loop별 표현력 차이를 보완합니다.
- `LTIInjection`은 recurrent update의 안정성을 관리합니다.
- `ACTHalting`은 position별로 언제 멈출지 예측합니다.

구조에서 목표가 되게 선명하게 드러나는 부분인데, parameter를 계속 늘리지 않고 inference-time compute를 늘리는 방향을 목표로 하는 것입니다.

일반 transformer에서 depth를 늘리려면 layer가 늘고 parameter도 늘어납니다. OpenMythos는 같은 block을 여러 번 돌려 effective depth를 늘립니다. 그래서 `n_loops`는 단순 옵션이 아니라 모델의 추론 깊이를 조절하는 knob으로 활용한 것입니다.

이것이 OpenMythos가 Mythos Preview를 설명하려는 핵심 방식으로 볼 수 있는데, Mythos Preview가 긴 작업을 잘 수행한다면 그 능력은 더 큰 모델 크기 때문일 수도, 더 좋은 post-training 때문일 수도, 더 강한 agent scaffold 때문일 수도 있습니다. OpenMythos는 그중 inference-time latent loop가 깊이를 만든다는 설명을 택해 구현한 것입니다.

---

## latent reasoning

OpenMythos에서 흥미로운 것 중 하나인데, latent reasoning을 모델 내부에 넣었습니다.

이 구조는 token으로 chain-of-thought를 출력하지 않고 같은 forward pass 안에서 hidden state가 반복 업데이트됩니다. 즉, intermediate thought가 텍스트로 남지 않습니다.

사실 이게 왜 흥미로운가 하면, 성능상으로 더 좋아지기 때문입니다. 물론 안전성과 안정성 관점에서는 조금 힘들어지는데, 토큰으로 출력되는 reasoning은 적어도 monitor가 읽을 수 있는 반면 hidden loop는 일반 transcript monitor가 볼 수 없습니다. 그래서 recurrent update가 안정적으로 유지되는지가 중요합니다.

OpenMythos는 이런 이슈를 `LTIInjection`으로 해결합니다.

```python
def get_A(self) -> torch.Tensor:
    return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

def forward(self, h, e, transformer_out):
    A = self.get_A()
    return A * h + self.B * e + transformer_out
```

원리가 되게 간단한데, `get_A()`는 항상 0과 1 사이의 값을 만듭니다. 이 값이 diagonal state matrix처럼 쓰입니다. 따라서 linear recurrence의 `A * h` 항만 보면 spectral radius가 1보다 작습니다.

이건 Parcae 계열의 안정화 관점으로, looped model은 반복이 깊어질수록 hidden state가 폭주할 수 있습니다. `A`를 안정 영역에 묶으면 그 위험을 줄일 수 있다는 것입니다.

다만, 실험적으로 nonlinear recurrent transformer의 안정성을 증명하진 않은 건 아쉬운 부분입니다. `transformer_out`은 attention과 MoE를 포함한 nonlinear term이기 때문에 이 항이 어떤 조건에서 bounded인지까지는 증명되지 않았습니다.

---

## Repeating the parameter reduces expressiveness

recurrent-depth 구조의 장점은 parameter 효율인데, 같은 block을 여러 번 쓰면 parameter 수를 늘리지 않고 depth를 늘릴 수 있기 때문입니다.

그런데 잠시 생각해보면 알 수 있듯이, 모든 loop가 같은 함수를 실행하면 첫 번째 loop와 마지막 loop가 다른 역할을 하기 어렵다는 게 단점입니다. 그래서 초기 탐색, 중간 조합, 마지막 검증이 같은 연산 형태로 뭉개질 수 있습니다.

OpenMythos는 이 문제를 두 단계로 다룹니다.

첫째, loop-index embedding을 더한다.

```python
angles = loop_t * freqs
emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
emb_full[:loop_dim] = emb
return h + emb_full.unsqueeze(0).unsqueeze(0)
```

> 레포 상에서 이름은 RoPE-like지만 실제 구현은 RoPE rotation이 아니라 sinusoidal bias가 좀 더 가까워 보입니다.

hidden state 앞쪽 일부 channel에 loop index signal을 더합니다. 같은 weight를 쓰더라도 현재 depth가 몇 번째인지 알 수 있게 만드는 장치입니다.

둘째, depth-wise LoRA를 쓴다.

```python
s = self.scale(torch.tensor(t_idx, device=x.device))
down = self.down(x) * s
return down @ self.B
```

큰 weight는 공유하지만, loop별 low-rank scale은 다르게 쓰는데 이 방식은 pure weight tying과 fully distinct layer 사이의 트레이드오프를 관리해줍니다.

두 개는 recurrent-depth의 표현력 손실을 줄이는 설계이긴 한데, `n_loops`가 `max_loop_iters`를 넘으면 LoRA는 마지막 scale을 재사용합니다. 훈련 범위 밖 loop에서 완전히 새로운 phase가 생기는 것은 아니지만, loop-index embedding은 계속 변하는 반면 LoRA adapter의 loop별 차이는 포화되기 때문에 extrapolation에는 한계가 있습니다. (그래서 이 부분에서 extrapolation을 보이고자 한다면 loop 수별 성능 곡선을 측정해볼 수 있습니다.)

---

## MoE

OpenMythos에서 parameter 대부분은 recurrent block의 MoE에 몰려 있습니다.

`TransformerBlock`은 `use_moe=True`일 때만 `MoEFFN`을 씁니다. Prelude와 Coda는 dense SwiGLU FFN입니다.

```python
self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
```

loop는 깊이를 만들고 MoE는 폭을 만듭니다. 같은 recurrent block을 반복하더라도, loop마다 hidden state가 바뀌면 router가 다른 expert를 선택할 수 있고, 그러면 같은 parameter set 안에서도 token과 loop depth에 따라 다른 sparse computation path가 생기게 됩니다.

Mythos Preview는 코딩, 브라우징, 장문 그래프 탐색, exploit, multimodal SWE, terminal task에서 강하다고 벤치에서 보여지는데, 이런 domain breadth를 설명하려면 넓은 지식 저장 경로가 필요하기 때문에 MoE를 사용해서 OpenMythos는 이걸 표현한 것 같습니다.

OpenMythos의 MoE 구현은 다음과 같습니다.

```python
logits = self.router(flat)
scores = F.softmax(logits, dim=-1)
_, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
topk_scores = scores.gather(-1, topk_idx)
topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
```

router는 expert 선택에는 `router_bias`를 더하고, 실제 gate weight는 bias 없는 `scores`에서 가져옵니다. 이 구조는 aux-loss-free routing 아이디어와 꽤 닮아 있습니다.

여기서 구현상의 의문점이 있는데, `router_bias`는 버퍼로 등록되어 있지만 학습 중 업데이트되는 코드가 없고, `training/3b_fine_web_edu.py`에도 expert load를 집계해 bias를 조정하는 로직이 없으며, main path에는 auxiliary load-balancing loss도 없어서 사실상 load-balanced MoE는 구현되어 있지 않습니다.

이게 딥러닝적으로 좀 중요한 이유는, MoE는 routing이 무너지면 sparse capacity가 무너지기 때문인데, 일부 expert만 선택되고 나머지가 죽으면 큰 parameter count는 실제 유효 용량으로 이어지지 않기 때문에 조심스레 여기서 언급해 봅니다.

---

## MLA

OpenMythos의 기본 attention은 MLA입니다.

MLA는 DeepSeek-V2에서 알려진 KV cache 절감 구조인데, full K/V를 cache하지 않고 latent representation을 저장한 뒤 필요할 때 K/V를 재구성하는 방법입니다.

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

cache에는 `c_kv`와 `k_rope`가 들어갑니다.

```python
kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}
```

그리고 attention 계산 시 `c_kv`에서 `k_nope`와 `v`를 재구성합니다.

다만 여기서 조금 의아한 점은 10–20배 memory reduction 했다는 README의 말인데, 현재 코드에서는 `k_rope`를 head 수만큼 expand한 뒤 cache하고, MLA cache 크기가 생각보다 커집니다. variant 기준으로 계산하면, 많은 설정에서 OpenMythos의 MLA cache는 이 코드의 GQA cache보다 오히려 큽니다.

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

계산한 표를 보면 OpenMythos의 MLA는 full MHA 대비 cache를 줄일 수 있습니다. 그러나 현재 구현과 프리셋만 보면 GQA 대비 항상 작지는 않습니다. (조심스레 필자의 생각을 밝히자면, README의 10–20배 감소와 코드 구현 사이에 괴리가 조금 있는 것 같습니다.)

---

## RoPE & generation cache

Mythos RoPE 구현을 보면 복소수 phasor를 미리 만들고, query/key의 인접 두 차원을 복소수로 본 뒤 position별 회전을 적용해서 norm을 보존하고 self-attention에서 relative position property를 만듭니다.

여기서 좀 중요한 지점은 `start_pos`인데, `OpenMythos.generate()`는 첫 step에서 prompt 전체를 넣고 이후 step에서는 마지막 token만 넣습니다. 이때 RoPE position이 계속 0으로 들어가면 cached key와 새 query의 상대 위치가 깨집니다.

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

(포스팅을 준비하며 깃헙 커밋 로그를 봤는데, 로그에도 RoPE decode position 버그를 고친 기록이 남아 있던 게 좀 재밌었습니다.)

---

## ACT

ACT는 Adaptive Computation Time으로, 각 position마다 halting probability를 예측합니다.

```python
return torch.sigmoid(self.halt(h)).squeeze(-1)
```

그리고 `RecurrentBlock`에서 누적 halting probability가 threshold를 넘으면 해당 position의 output 기여를 멈춥니다.

이런 설계는 긴 context에서 의미 있는데, 긴 context의 모든 token이 다른 처리 난이도를 가집니다. 어떤 token은 문법적 연결만 필요하고, 어떤 token은 멀리 떨어진 코드 의존성이나 수학 조건을 조합해야 합니다. 모든 position에 동일한 loop 수를 주면 compute 낭비가 생기지만, ACT는 position별 compute allocation을 가능하게 합니다.

---

## Tokenizer & training script

```python
DEFAULT_MODEL_ID = "openai/gpt-oss-20b"

class MythosTokenizer:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
```

학습 스크립트는 FineWeb-Edu streaming dataset, FSDP, AdamW, mixed precision, checkpointing을 할 수 있게 만들어뒀습니다.

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

다만 pretrain을 할 때 과정에서 얻어지는 정보가 좀 부족합니다.

- validation loop가 없음
- benchmark evaluation이 없음
- MoE expert utilization logging이 없음
- ACT 평균 loop count logging이 없음
- hidden norm by loop가 없음
- router load balancing update가 없음
- streaming dataset resume position이 복원되지 않음

특히 MoE와 ACT를 넣은 모델은 내부 metric이 없으면 학습 실패 원인을 알기 어려운데, loss만 보고는 expert collapse인지, halting collapse인지, recurrent norm 문제인지 구분할 수 없다는 부분이 있습니다.

---

## Conclusion: OpenMythos's Value

OpenMythos 레포는 Mythos System Card가 보여준 장기 작업 능력을 하나의 계산 모델로 설명하고 있습니다.

- 깊이는 더 많은 layer stack이 아니라 recurrent loop로 산다.
- 폭은 recurrent loop 안의 MoE expert pool로 확보한다.
- 안정성은 input reinjection, LTI-stable update, loop-index signal, ACT로 관리하려 한다.

이 관점에서 OpenMythos는 "장기 agentic reasoning을 모델 내부 반복 계산으로 설명할 수 있는가"라는 질문을 꽤 코드로 잘 짠 레포입니다.

물론 완성된 훈련 셋은 아니고, 검증되어 우리가 쓸 수 있는 Offensive model도 아니긴 합니다.

그럼에도 읽을 가치가 충분한데, 앞으로 고능한 coding agent와 cyber agent들 자체가 단순 parameter 수만으로 설명되지 않기 때문입니다.

최근 여러 소식을 통해 스스로 Task force한 Local Agent나 Local AI를 만들고자 하는 분들이 많다는 것을 듣습니다.

그렇기에 이런 레포를 통해 더 많은 inference-time compute를 어디에 쓰는지, token으로 보이지 않는 latent reasoning을 어떻게 안정화하는지, 긴 작업에서 hidden state가 어떻게 drift하지 않게 만드는지, safety monitor가 hidden loop를 어디까지 볼 수 있는지를 알고 있다면 Task force한 AI를 만드는 과정에서 유용하게 쓸 수 있을 거라는 개인적인 생각을 전하며 글을 마치겠습니다.

(긴 글인데 읽어주셔서 감사합니다. 열심히 배우고 성장하는 중이오니, 혹시 글 내용에서 실수하거나 부족해 보이는 점에 대한 피드백은 devmhyun@gmail.com 으로 보내주시면 열심히 배우겠습니다. 다시 한 번 읽어주셔서 감사합니다.)

---

## 참고한 자료들

- Anthropic, [Assessing Claude Mythos Preview's cybersecurity capabilities](https://red.anthropic.com/2026/mythos-preview/)
- Anthropic, [Claude Mythos Preview System Card PDF](https://www-cdn.anthropic.com/8b8380204f74670be75e81c820ca8dda846ab289.pdf)
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
