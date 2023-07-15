# KlangGPT
base on FinGPT
基于FinGPT的金融问答系统，首期数据来自 FinGPT项目。

通过获取网络数据，将数据训练到现在有的大模型里（ChatGLM-6B）。

训练后的模型 可以进行问答，数据可以持续更新训练。

平价模型，采用平价模型让大家都可以轻松训练更新数据。

# 加入群
<img src=figs/qrcode.jpeg width=25% />

# data 
金融数据来自 FinGPT

# train.py 
个性化训练
来自 https://github.com/lich99/ChatGLM-finetune-LoRA

# 使用例子 infer.py 

```
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
```
 
