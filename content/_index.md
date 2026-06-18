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

## Experiences

{{< entry date="Jul 2023 – Present" title="Undergraduate Researcher" org="**CSEC Lab. Soongsil Univ.**" >}}
- AI Safety Researcher
  - Research on interpreting LLM representations through mechanistic interpretability
  - Research on AINN attack&defense
  - Researching red teaming techniques such as adversarial prompts and jail breaks to penetrate actual LLM services, targeting LLMs
{{< /entry >}}

{{< entry date="Nov 2025 – Jan 2026" title="ML Engineer" org="**Microsoft Singapore**(Outsourcing)" >}}
- ML Engineer for an AI model-based service demo optimized for Microsoft laptop NPUs (for Microsoft client showcases)
{{< /entry >}}

{{< entry date="Jun 2025 - Aug 2025" title="Operator" org="**ACSC2025** (OutSourcing)" >}}
- Operated two AI security problems to ASCS2025, the Asian qualifier for the ICC World Hacking Defense Competition.
  - Vulnerabilities in TPM-based encrypted communication methods utilizing Neural Network
  - Reverse engineering a VM implemented as a neural network using an ONNX file format model.
{{< /entry >}}

{{< entry date="Dec 2024" title="Instructor" org="**DAY-1 Company** (OutSourcing)" >}}
- Participated as an instructor in the 25 Python Skill-up Data Analysis MASTER CLASS lectures.
{{< /entry >}}

## CVE
{{< cve id="CVE-2026-5241" tech="Huggingface" icon="https://cdn.simpleicons.org/huggingface/transformers" type="Remote Code Execution" cvss="v3.1" score="9.6" level="Critical" title="Arbitrary code execution via nested trust_remote_code bypass" nvd="https://nvd.nist.gov/vuln/detail/CVE-2026-5241">}}
{{< /cve >}}

{{< cve id="CVE-2025-66960" tech="Ollama" icon="https://cdn.simpleicons.org/ollama" type="Denial of Service" cvss="v3" score="7.5" level="High" title="GGUF v1 string length panics readGGUFV1String" link="/blog/cve-2025-66960guf-v1-string-length-cause-panic-in-readggufv1string/" nvd="https://nvd.nist.gov/vuln/detail/CVE-2025-66960" >}}
A malformed v1 string length is used without bounds validation, panicking `readGGUFV1String` during model creation.
{{< /cve >}}

{{< cve id="CVE-2025-66959" tech="Ollama" icon="https://cdn.simpleicons.org/ollama" type="Denial of Service" cvss="v3" score="7.5" level="High" title="Unchecked length in the GGUF decoder causes a remote panic / DoS" link="/blog/cve-2025-66959panic-dos-via-unchecked-length-in-gguf-decoder-copy/" nvd="https://nvd.nist.gov/vuln/detail/CVE-2025-66959" >}}
Creating a model from a crafted GGUF blob makes `readGGUFString` allocate a slice from an unchecked 8-byte length, panicking with `makeslice: len out of range` and crashing the server.
{{< /cve >}}

## Honours & Awards

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

{{< entry date="Dec 21, 2025" title="블루오션인줄 알고 뛰어든 인공지능보안에 대하여" org="**HolyShield 2025**" >}}
{{< /entry >}}


## Education

{{< entry date="2023 – present" title="Soongsil University" org="Software Engineering">}}
{{< /entry >}}