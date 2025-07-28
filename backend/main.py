import modal

app = modal.App("Liora-ai")


image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /temp/ACE-STEP", "cd /temp/ACE-STEP && pip install ."])
    .env({"HFM_FOLDER" : "/.cache/huggingfacemodels"})
    .add_local_python_source("prompts")
)


ace_model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
## we giving a storage now to store the ace model 
hf_volume = modal.Volume.from_name("qwen-hf-model", create_if_missing=True)

Liora_ai_secret = modal.Secret.from_name("custom-secret")


@app.cls(
    image= image,
    gpu="L40S",
    volumes={"/music-model" : ace_model_volume, "/.cache/huggingfacemodels" : hf_volume},
    secrets=[Liora_ai_secret],
    scaledown_window=15
)
class MusicServer:
    @modal.enter()
    def start_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        # from transformers import AutoModelForCausalLM, AutoTokenizer

        # generating music generation model 
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/music-model",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )


        
        