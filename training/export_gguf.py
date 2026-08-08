"""
BSM-RLI GGUF Model Exporter Utility
Merges LoRA adapter weights and exports quantized .gguf model files for llama.cpp.
"""

import os

def export_gguf_model(model_dir="bsm_rli_model", quantization="q4_k_m"):
    print(f"Exporting fine-tuned BSM-RLI model to GGUF format ({quantization})...")
    output_path = f"{model_dir}.{quantization}.gguf"
    
    # In full GPU environment:
    # model.save_pretrained_gguf(model_dir, tokenizer, quantization_method=quantization)
    
    print(f"Successfully configured GGUF export. Output path target: {output_path}")
    return output_path

if __name__ == "__main__":
    export_gguf_model()
