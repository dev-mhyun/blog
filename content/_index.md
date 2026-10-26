---
name: "Myeonghyun Kim"
role: "ML/DL Engineer & AISec Researcher"
bio: "Hi, I'm Myeonghyun Kim(@kimh). I'm really into computers, specifically focusing on AI, AI security, and offensive security. I enjoy doing CTFs and bug bounties in my free time, and I'm also researching AI and AI security."
photo: "/img/me.jpg"
email: "devmhyun@gmail.com"
github: "https://github.com/dev-mhyun"
blog: "https://zero.shotlearni.ng/blog/"
x : "https://x.com/desckimh"
cv: "/cv/"
---

## Education

{{< entry date="2023 – present" title="B.S., Computer Science and Software Engineering" org="Soongsil University, Seoul, Korea">}}
{{< /entry >}}

## Experiences

{{< entry date="Jul 2023 – Present" title="Undergraduate Researcher" org="[**CSEC Lab. Soongsil Univ.**](http://csec.ssu.ac.kr/)" >}}
- AI Safety Researcher
  - Research on interpreting LLM representations through mechanistic interpretability
  - Research on AINN attack&defense
  - Researching red teaming techniques such as adversarial prompts and jail breaks to penetrate actual LLM services, targeting LLMs
{{< /entry >}}

{{< entry date="Oct 2026 – Present" title="Researcher&Engineer" org="[**STEALIEN**](https://stealien.com/)" >}}
- NDA
{{< /entry >}}

{{< entry date="Nov 2025 – Jan 2026" title="ML Engineer" org="[**Microsoft Singapore**](https://local.microsoft.com/communities/asia-pacific/singapore/) (Outsourcing)" >}}
- ML Engineer for an AI model-based service demo optimized for Microsoft laptop NPUs (for Microsoft client showcases)
{{< /entry >}}

{{< entry date="Jun 2025 – Aug 2025" title="Operator" org="[**ACSC2025**](https://acsc.asia/) (OutSourcing)" >}}
- Operated two AI security problems to ASCS2025, the Asian qualifier for the ICC World Hacking Defense Competition.
  - Vulnerabilities in TPM-based encrypted communication methods utilizing Neural Network
  - Reverse engineering a VM implemented as a neural network using an ONNX file format model.
{{< /entry >}}

## Publications

{{< pub
  venue="EMNLP(Main)" year="2026" type="conference" status="to appear"
  title="Legible but Not Localized: Jailbreak Geometry in Mixture-of-Experts Language Models Is Attack-Specific"
  authors="Yubin IM, Myeonghyun Kim, Seongmin Kim, and Haehyun Cho"
  full="The 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)"
  location="Budapest, Hungary" date="Oct 2026" >}}
{{< /pub >}}

## CVEs & Bounties

{{< cve tech="Kantega SSO" icon="/img/kantega-sso-logo.png" type="Authentication Bypass" sev="P1" bounty="1,500" >}}
{{< /cve >}}

{{< cve tech="Atlassian-Built Apps" icon="https://cdn.simpleicons.org/atlassian" type="Broken Authentication and Session Management" sev="P2" bounty="1,000" >}}
{{< /cve >}}

{{< cve id="NBB-2026-0263" tech="Naver" icon="https://cdn.simpleicons.org/naver" type="Cross Origin DOM XSS" cvss="v4" score="N/A" bounty="200" title="Chzzk Account Takeover and PII Leakage via Cross-Origin DOM-Based XSS">}}
{{< /cve >}}

{{< cve id="HVE-2026-0022" tech="Hspace" icon="/img/hspace-logo.png" type="Account Takeover" sev="Critical" bounty="1,200" title="Account Takeover via Unverified Refresh Token Signature">}}
{{< /cve >}}

{{< cve id="CVE-2026-5241" tech="Huggingface" icon="https://cdn.simpleicons.org/huggingface/transformers" type="Remote Code Execution" cvss="v3.1"  bounty="750" score="9.6" level="Critical" title="Arbitrary code execution via nested trust_remote_code bypass" nvd="https://nvd.nist.gov/vuln/detail/CVE-2026-5241">}}
{{< /cve >}}

{{< cve id="CVE-2025-66960" tech="Ollama" icon="https://cdn.simpleicons.org/ollama" type="Denial of Service" cvss="v3" score="7.5" level="High" title="GGUF v1 string length panics readGGUFV1String" link="/blog/cve-2025-66960guf-v1-string-length-cause-panic-in-readggufv1string/" nvd="https://nvd.nist.gov/vuln/detail/CVE-2025-66960" >}}
A malformed v1 string length is used without bounds validation, panicking `readGGUFV1String` during model creation.
{{< /cve >}}

{{< cve id="CVE-2025-66959" tech="Ollama" icon="https://cdn.simpleicons.org/ollama" type="Denial of Service" cvss="v3" score="7.5" level="High" title="Unchecked length in the GGUF decoder causes a remote panic / DoS" link="/blog/cve-2025-66959panic-dos-via-unchecked-length-in-gguf-decoder-copy/" nvd="https://nvd.nist.gov/vuln/detail/CVE-2025-66959" >}}
Creating a model from a crafted GGUF blob makes `readGGUFString` allocate a slice from an unchecked 8-byte length, panicking with `makeslice: len out of range` and crashing the server.
{{< /cve >}}

## Honours & Awards

{{< ctfs title="CTFs" >}}
  {{< ctf event="2026 KISIA Information Security Competition" tier="gold" award="Honorable Mention" org="Ministry of Science and ICT (MSIT), Korea Information Security Industry Association (KISIA)" date="Aug 2026" link="https://www.dailysecu.com/news/articleView.html?idxno=208122" >}}
  {{< ctf event="HackTheon Sejong International Collegiate Cybersecurity Competition 2026" tier="gold" award="Excellence Award (Advanced Division)" org="Sejong Special Self-Governing City, National Security Research Institute" date="July 2026" link="https://v.daum.net/v/20260710155058437" >}}
  {{< ctf event="UMDCTF 2026" rank="1st" tier="gold" >}}
  {{< ctf event="Vishwa CTF 2026" rank="1st" tier="gold" >}}
  {{< ctf event="Silly CTF 2026" rank="1st" tier="gold" >}}
  {{< ctf event="Blue Hens CTF 2026" rank="1st" tier="gold" >}}
  {{< ctf event="Break The Syntax CTF 2026" rank="1st" tier="gold" >}}
  {{< ctf event="squ1rrel CTF 2026" rank="1st" tier="gold" award="Open division" >}}
  {{< ctf event="b01lers CTF 2026" rank="2nd" tier="silver" award="Jeopardy 1st place" >}}
  {{< ctf event="Kalmar CTF 2026" rank="2nd" tier="silver" >}}
  {{< ctf event="Ping CTF 2026" rank="2nd" tier="silver" >}}
  {{< ctf event="TexSaw CTF 2026" rank="2nd" tier="silver" >}}
  {{< ctf event="Dawg CTF 2026" rank="2nd" tier="silver" >}}
  {{< ctf event="UndutManing CTF 2026" rank="2nd" tier="silver" >}}
  {{< ctf event="Midnight Flag CTF 2026" rank="3rd" tier="bronze" >}}
  {{< ctf event="UMass CTF 2026" rank="3rd" tier="bronze" >}}
  {{< ctf event="TJCTF 2026" rank="3rd" tier="bronze" >}}
  {{< ctf event="BSidesSF CTF 2026" rank="3rd" tier="bronze" >}}
  {{< ctf event="UnivsThreats CTF 2026 Quals" rank="3rd" tier="bronze" >}}
  {{< ctf event="Batman's Kitchen CTF 2026" rank="3rd" tier="bronze" >}}
  {{< ctf event="EhaX CTF 2026" rank="4th" >}}
  {{< ctf event="UTCTF 2026" rank="4th" >}}
  {{< ctf event="TAMU CTF 2026" rank="4th" >}}
  {{< ctf event="Srdnlen CTF 2026 Quals" rank="5th" >}}
  {{< ctf event="Codegate CTF 2026 Quals" rank="5th" >}}
  {{< ctf event="DICE CTF 2026 Quals" rank="6th" award="Finalist" >}}
  {{< ctf event="Midnight Sun CTF Quals" rank="6th" >}}
  {{< ctf event="BITS CTF 2026" rank="6th" >}}
{{< /ctfs >}}

{{< award title="NAVER Hall of Fame" category="Bug bounty" catcolor="#d4af37" place="🏆" tier="gold" org="Naver" date="2026" link="https://bugbounty.naver.com/hall-of-fame" >}}
NAVER Hall of Fame 2026
{{< /award >}}

{{< award title="Hspace Hall of Fame" category="Bug bounty" catcolor="#d4af37" place="🏆" tier="gold" org="Hpsace" date="2026" link="https://hspace.io/hall-of-fame" >}}
Hspace Hall of Fame 2026 1st.
{{< /award >}}

{{< award title="2025 자랑스런 소프트웨어인상" category="University" catcolor="#00567d" place="🏆" tier="gold" org="Soongsil Univ. Software" date="Sep 2025" >}}
Awards for outstanding undergraduate students in 2025.
{{< /award >}}

{{< award title="Best Paper Award" category="Paper" catcolor="#2f7d6b" place="🏆" tier="gold" org="ASK 2025" date="May 2025" >}}
Awarded for "얼굴형 분석 기반 헤어스타일링 추천 서비스".
{{< /award >}}

{{< award title="Best Paper Award" category="Paper" catcolor="#2f7d6b" place="🏆" tier="gold" org="CISC-S'24" date="Jun 2024" >}}
Awarded for "악성 파일 탐지 모델 취약성 분석 및 방어 프레임워크".
{{< /award >}}


## Speaker

{{< entry date="Aug 23, 2026" title="!safetensors, un?trust_remote_code" org="**HakingCamp 33rd**" >}}
{{< /entry >}}
{{< entry date="Dec 21, 2025" title="블루오션인줄 알고 뛰어든 인공지능보안에 대하여" org="**HolyShield 2025**" >}}
{{< /entry >}}