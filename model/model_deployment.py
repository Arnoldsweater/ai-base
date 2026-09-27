from modelscope import snapshot_download

model_dir1 = snapshot_download(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="./llm/Qwen/Qwen2.5‑0.5B‑Instruct"
)

model_dir2 = snapshot_download(
    model_id="Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",
    local_dir="./llm/Qwen/Qwen1.5‑1.8B‑Chat‑GPTQ‑Int4"
)

print("Qwen0.5B保存路径：", model_dir1)
print("Qwen1.8B保存路径：", model_dir2)
