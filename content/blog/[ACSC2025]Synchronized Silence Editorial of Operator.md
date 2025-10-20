## **TL; DR**

Tree Parity Machine(TPM)을 이용해 AES 키를 공개 채널에서 학습 동기화로 생성한 뒤, 암호화된 플래그를 제공하는 문제다. 로그(log.json)를 기반으로 TPM weight를 재현하고, SHA256으로 키를 유도해 enc_flag.txt를 복호화하면 된다. naive한 단일 TPM imitation으로는 풀리지 않으며,  population attack, mutation, selective unit flipping을 활용해 풀이 가능하다.

## **문제 구조**

- log.json: TPM 학습 로그. 각 step의 입력 벡터(x: 3x100)와 출력 bit(tau: {-1, +1})가 주어짐. 단, Alice와 Bob이 tau^A = tau^B일 때만 기록됨.
- enc_flag.txt: AES-ECB로 암호화된 플래그.
- implementation.py : 구조체
- README.md : 걍 시나리오, 목표

## **출제의도**

TPM은 neural cryptography에서 양측이 키를 공유하는 방식이다. 학습 기반 키 교환이 비동기적으로 이루어지고, 공개된 채널에서 이뤄진다는 점이 매력 적이지만, 공격자의 입장에서 동기화된 키를 재현가능 함.

해당 문제는 4가지의 근복적 취약점이 있다

1. 출력 공개
2. 동기식 업데이트 (weight 경로의 결정성)
3. 상태공간의 유한성
4. 학습 규칙의 공개

## 취약점 분석

### **1. Tree Parity Machine(TPM)의 형식적 정의**

- **구조**
    - 은닉층 뉴런 K=3
    - 각 뉴런 입력 차원 N=100
    - 가중치 범위 $w_{k,j}\in\{-L,\dots,L\},\;L=3$
- **국소장(local field)**
    
    $h_k(t)=\sum_{j=1}^{N}w_{k,j}(t)\,x_{k,j}(t),\qquad
    \sigma_k(t)=\operatorname{sgn}\!\bigl(h_k(t)\bigr)$
    
    (0은 +1로 치환)
    
- **출력 비트**
    
    $\tau(t)=\prod_{k=1}^{K}\sigma_k(t)\in\{-1,+1\}$
    
    이 값만이 공개 채널을 통해 누설된다.
    
- **Hebbian 학습 규칙**
    
    $w_{k,j}(t+1)=g\!\Bigl(w_{k,j}(t)+\sigma_k(t)\,x_{k,j}(t)\,\Theta\!\bigl(\sigma_k(t)\tau(t)\bigr)\Bigr)$
    
    여기서 $g(\cdot)$는 범위를 자르는 함수이고 $\Theta$는 부호 일치 여부(0/1) 지시자다 .
    

---

### **2. 출력 공개**

공격자가 시퀀스

$\bigl\{x(t),\tau(t)\bigr\}{t=1}^{T}$

를 얻을 때, 한 스텝에서 숨겨진 $\sigma_k$ 3비트 중 알려지는 정보량은 **1 bit**뿐이다.

*엔트로피 관점에서*

$H\bigl(\sigma_1,\sigma_2,\sigma_3\bigr)=3
,\qquad
H\bigl(\sigma_1,\sigma_2,\sigma_3\mid\tau)=2$

즉 **한 스텝당 1 bit**가 그대로 남는다. 스텝의 로그는

$I{\text{leak}} = T\text{ bit}$

을 노출하며, 이는 가중치 공간

$(2L+1)^{KN}=7^{300}$

에 비하면 작지만 selection-filtering 과 결합될 때 치명적이다.

---

### **3. Population Attack**

1. **초기화** $n_0$개의 후보 가중치 벡터 $\{W_i^{(0)}\}$를 무작위로 뽑음
2. **필터 단계**
    
    $P_{t+1}= \Bigl\{\,g\bigl(W_i^{(t)},x(t),\tau(t)\bigr)\;|\;\tau\bigl(W_i^{(t)},x(t)\bigr)=\tau(t)\Bigr\}$
    
    여기서 g는 Hebbian update 연산.
    
3. **크기 기대값**
    
    각 스텝의 생존 확률은
    
    $p_{\text{match}}(t)=\Pr\bigl[\tau(W,x)=\tau(t)\bigr]\approx\frac12\bigl(1+\epsilon_t\bigr)$
    
    (무작위 가중치면 $\epsilon_t\approx0$)
    
    따라서
    
    $\mathbb E\!\bigl[|\mathcal P_{t}|\bigr]=n_0\,\prod_{s=1}^{t-1}p_{\text{match}}(s)$
    
    로그 길이가 충분하면 $|\mathcal P_{t}|\to0$ → 실패. 이를 **선택적 플립**과 **돌연변이**로 보완
    

---

### **4. Selective Unit Flipping**

- 공격자가 $\tau_{\text{pred}}\neq\tau$인 스텝에서
    
    $k^\ast = \arg\min_k \left| h_k \right|$
    
    을 찾아 $\sigma_{k^\ast} \leftarrow -\sigma_{k^\ast}$ 로 강제 전환하면 새 출력 **$\tau’ = -\tau_{\text{pred}}$
    
    즉 정확히 한 비트 반전으로 불일치 해소가 가능할 확률은
    
    $P_{\text{flip}} = \Pr\left[ \operatorname{sgn}(h_{k^\ast}) \neq 0 \right] \approx 1$
    
    (국소장이 0이면 이미 $\tau$ 일치였음).
    
- 필터 손실을 $O(1)$로 줄여 $|\mathcal P_t|$ 붕괴를 방지 가능함

---

### **5. Mutation Strategy**

필터+플립에도 집단 다양성이 급감하면 탐색이 지수적으로 느려진다.

이를 해결하기 위해 남은 개체 W마다

$W’\;=\;W+\Delta W,\qquad
\Delta W_{k,j}\sim\{-1,0,+1\}$

를 확률 $p_m$로 삽입, **전이 행렬** M을 두면

$\mathbf n_{t+1}= \mathbf n_t P_{\text{filter}} + \mathbf n_t M$

M이 **가중치 랜덤 워크**를 부여하여 마코프 체인의 미소 공간 접근성을 보장

---

### **6. 공격 성공 확률 & 시간 복잡도**

$P_{\text{sync}}^{\text{Eve}}(T)\approx1-\exp\!\bigl(-\alpha\,T/n_0\bigr)$, $\alpha\propto\frac{1}{(2L+1)^N}$

동일 파라미터에서 **50 k 개체, 10 k 스텝**이면 $P_{\text{sync}}\to0.9$ 수준

연산량은 GPU 없이도 수 분–수십 분 내 실행 가능

$O(n_0 K N T)\approx 5\!\times\!10^4\cdot3\cdot100\cdot10^4\approx1.5\!\times\!10^{9}$

---

### **7. 키 파생 공식**

가중치 행렬 W를 시프트 & 3진법 인코딩 → 비트열 $b$ →

$K_{\text{AES}}=\operatorname{Trunc}_{128}\!\Bigl(\operatorname{SHA256}\bigl(\mathbf b\bigr)\Bigr)$

이는 **가중치 1 bit라도 틀리면 해시가 전혀 달라지므로** 공격자에게 완전체 동기화를 강제.

## 솔버

위 수식들 기반으로 솔버 작성하면

```python
import json
import numpy as np
from Crypto.Cipher import AES
import hashlib
import random

K, N, L = 3, 100, 3
psize = 50000
b = [-1, 0, 1]
mthreshold = 100
mfactor = psize // 5

def tpm_output(w, x):
    dot = np.einsum('ij,ij->i', w, x)
    sig = np.sign(dot)
    sig[sig == 0] = 1
    tau = np.prod(sig)
    return tau, sig, dot

def update_weights(w, x, sig, tau):
    for i in range(K):
        if sig[i] == tau:
            w[i] += x[i]
            w[i] = np.clip(w[i], -L, L)
    return w

def weight_to_aes_key(w):
    bits = ''.join(f'{(v + L):03b}' for v in w.flatten())
    b = int(bits, 2).to_bytes((len(bits) + 7) // 8, byteorder='big')
    return hashlib.sha256(b).digest()[:16]

def load():
    with open("log.json") as f:
        log = json.load(f)
    with open("enc_flag.txt", "rb") as f:
        ct = f.read()
    return log, ct

def biased_init():
    return np.random.choice(b, size=(K, N))

def mutate(w, strength=1):
    mutation = np.random.randint(-strength, strength + 1, size=w.shape)
    return np.clip(w + mutation, -L, L)

def population_attack():
    log, ct = load()
    population = [biased_init() for _ in range(psize)]

    for t, entry in enumerate(log):
        x = np.array(entry['x'])
        tau_true = entry['tau']
        new_pop = []

        for w in population:
            tau_pred, sig, dot = tpm_output(w, x)
            if tau_pred == tau_true:
                new_pop.append(update_weights(w.copy(), x, sig, tau_true))
            else:
                i_star = np.argmin(np.abs(dot))
                sig[i_star] *= -1
                tau_flip = np.prod(sig)
                if tau_flip == tau_true:
                    new_pop.append(update_weights(w.copy(), x, sig, tau_true))

        population = new_pop
        print(f"[Step {t:03d}] Population: {len(population)}")

        if len(population) == 0:
            print("[-] nono ")
            return

        if len(population) < mthreshold:
            print(f"[*] Low survivors ({len(population)}), mutating...")
            survivors = population.copy()
            population = []
            for w in survivors:
                for _ in range(mfactor // max(1, len(survivors))):
                    population.append(mutate(w, strength=1))
            print(f"[*]New population after mutation: {len(population)}")

    print(f"[*]{len(population)} survivors. Trying AES decryption...")
    for w in population:
        key = weight_to_aes_key(w)
        pt = AES.new(key, AES.MODE_ECB).decrypt(ct)
        if b"acsc{" in pt:
            print(pt.decode(errors="ignore"))
            return

if __name__ == "__main__":
    population_attack()

```