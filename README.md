# LongRoPE

It's unofficial implementation of LongRoPE for Llama3. 

It's similar as Yarn or RoPE NTK method to support long context LLM However, it is additional beneficial to allow 8 × extension without additional fine-tuning.
that means we could use 64K tokens 8K x 8 = 64K tokens without fine-tuning. So I implement it for evaluate LongRoPE without fine-tune. Authors already mentioned the official site but the official Code is not released yet(24 May2024 ) please check their offical site for official implementation. 
more detail information, check the paper [LongRope :Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753)


### Abstract 
Large context window is a desirable feature in large language models (LLMs). However, due to high fine-tuning costs, scarcity of long texts, and catastrophic values introduced by new token positions, current extended context windows are limited to around 128k tokens.

This paper introduces LongRoPE that, for the first time, extends the context window of pre-trained LLMs to an impressive 2048k tokens, with up to only 1k fine-tuning steps at within 256k training lengths, while maintaining performance at the original short context window. This is achieved by three key innovations: (i) we identify and exploit two forms of non-uniformities in positional interpolation through an efficient search, providing a better initialization for fine-tuning and enabling an 8
×
 extension in non-fine-tuning scenarios; (ii) we introduce a progressive extension strategy that first fine-tunes a 256k length LLM and then conducts a second positional interpolation on the fine-tuned extended LLM to achieve a 2048k context window; (iii) we readjust LongRoPE on 8k length to recover the short context window performance. Extensive experiments on LLaMA2 and Mistral across various tasks demonstrate the effectiveness of our method. Models extended via LongRoPE retain the original architecture with minor modifications to the positional embedding, and can reuse most pre-existing optimizations. Code will be available at [https://github.com/microsoft/LongRoPE](https://github.com/microsoft/LongRoPE) 

 
## code structure 
- `long_rope` : longRoPE implementation
- `LongRoPEWrapper` : huggingface transformer wrapper 

