from pathlib import Path
from typing import Any, List, Optional

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from config.settings import (
    DEVICE,
    DO_SAMPLE,
    EMBEDDING_MODEL_PATH,
    LLM_MODEL_PATH,
    MAX_NEW_TOKENS,
    RERANKER_MODEL_PATH,
    TEMPERATURE,
)


def load_embeddings() -> HuggingFaceEmbeddings:
    """加载本地 BGE 中文嵌入模型（bge 系列需归一化）。"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


class LocalQwenLLM(LLM):
    """基于本地 Qwen 权重的对话式 LLM 封装。

    直接把裸文本喂给 text-generation pipeline 时，模型会把 prompt 当成
    "续写开头"照抄一遍（表现为答案里回显系统提示和上下文）。
    这里先把 prompt 包成标准的 user 消息、套上 tokenizer 自带的 chat 模板，
    再让模型生成，从而得到干净的回答。
    """

    pipeline: Any = None
    tokenizer: Any = None
    max_new_tokens: int = MAX_NEW_TOKENS

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "local-qwen-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipeline(
            text,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"]

        # 兜底：个别版本会忽略 return_full_text，仍返回完整文本
        if isinstance(generated, str) and generated.startswith(text):
            generated = generated[len(text):]

        answer = generated.strip()

        if stop:
            for token in stop:
                idx = answer.find(token)
                if idx != -1:
                    answer = answer[:idx]
        return answer


def load_llm() -> LocalQwenLLM:
    """从本地 Qwen 加载文本生成 LLM。"""
    path = Path(LLM_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    # 避免生成时出现 pad_token 未设置的警告
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": 1.1,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **gen_kwargs,
    )

    return LocalQwenLLM(
        pipeline=pipe,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
    )


class LocalReranker:
    """本地 BGE 交叉编码器重排模型。"""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_PATH, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(DEVICE)
        self.model.eval()

    def rerank(self, query: str, docs: list, top_k: int = 3) -> list:
        pairs = [(query, d.page_content) for d in docs]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]
