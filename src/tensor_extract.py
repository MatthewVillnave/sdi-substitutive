#!/usr/bin/env python3
"""
tensor_extract.py — Extract attn_out layer 0 from GGUF files.
Supports real extraction via gguf package.
"""

import sys
import os
import argparse
import json
import numpy as np

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(__file__))

GGUF_Q4 = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q4_k_m.gguf"
GGUF_Q3 = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q3_k_m.gguf"
GGUF_Q2 = "/media/matthew-villnave/VL_usb/models/qwen2.5-0.5b-official/qwen2.5-0.5b-instruct-q2_k.gguf"

def extract_attn_output(gguf_path: str) -> dict:
    """Extract attn_output layer weights from GGUF file."""
    try:
        import gguf
    except ImportError:
        return {"status": "blocked", "reason": "gguf package not installed"}
    
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception as e:
        return {"status": "blocked", "reason": f"GGUFReader failed: {e}"}
    
    # Find blk.0.attn_output.weight
    for t in reader.tensors:
        if t.name == "blk.0.attn_output.weight":
            try:
                w = gguf.dequantize(t.data, t.tensor_type)
                return {
                    "status": "extracted",
                    "weights": w,
                    "shape": list(w.shape),
                    "dtype": str(w.dtype),
                    "tensor_type": t.tensor_type,
                    "n_elements": t.n_elements,
                    "n_bytes": t.n_bytes,
                    "gguf_path": gguf_path,
                }
            except Exception as e:
                return {"status": "blocked", "reason": f"dequantize failed: {e}"}
    
    return {"status": "blocked", "reason": "blk.0.attn_output.weight not found"}


def main():
    parser = argparse.ArgumentParser(description="Extract attn_out layer 0 from GGUF")
    parser.add_argument("--gguf-q4", default=GGUF_Q4)
    parser.add_argument("--gguf-q3", default=GGUF_Q3)
    parser.add_argument("--gguf-q2", default=GGUF_Q2)
    parser.add_argument("--out-dir", default=os.path.dirname(__file__) + "/../results")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print("Extracting attn_out layer 0 from Q4 model...")
    result_q4 = extract_attn_output(args.gguf_q4)
    
    result = {
        "tensor_extracted": result_q4["status"] == "extracted",
        "extraction_details": {},
    }
    
    if result_q4["status"] != "extracted":
        result["classification"] = "BLOCKED_TENSOR_EXTRACTION"
        result["block_reason"] = result_q4.get("reason", "unknown")
        print(json.dumps(result, indent=2, default=str))
        return
    
    # Extract Q3 and Q2 as well
    W_ref = result_q4["weights"]
    result["extraction_details"] = {
        "q4": {
            "shape": result_q4["shape"],
            "dtype": result_q4["dtype"],
            "tensor_type": result_q4["tensor_type"],
            "n_elements": result_q4["n_elements"],
            "gguf_path": args.gguf_q4,
        }
    }
    
    print("Extracting Q3 model...")
    result_q3 = extract_attn_output(args.gguf_q3)
    if result_q3["status"] == "extracted":
        result["extraction_details"]["q3"] = {
            "shape": result_q3["shape"],
            "dtype": result_q3["dtype"],
            "tensor_type": result_q3["tensor_type"],
        }
        W_q3 = result_q3["weights"]
    
    print("Extracting Q2 model...")
    result_q2 = extract_attn_output(args.gguf_q2)
    if result_q2["status"] == "extracted":
        result["extraction_details"]["q2"] = {
            "shape": result_q2["shape"],
            "dtype": result_q2["dtype"],
            "tensor_type": result_q2["tensor_type"],
        }
        W_q2 = result_q2["weights"]
    
    # Save arrays to results dir (as npy)
    np.save(os.path.join(out_dir, "W_q4_attn_out_L0.npy"), W_ref)
    result["files_written"] = {
        "W_q4_attn_out_L0.npy": os.path.getsize(os.path.join(out_dir, "W_q4_attn_out_L0.npy"))
    }
    
    if result_q3["status"] == "extracted":
        np.save(os.path.join(out_dir, "W_q3_attn_out_L0.npy"), W_q3)
        result["files_written"]["W_q3_attn_out_L0.npy"] = os.path.getsize(os.path.join(out_dir, "W_q3_attn_out_L0.npy"))
    
    if result_q2["status"] == "extracted":
        np.save(os.path.join(out_dir, "W_q2_attn_out_L0.npy"), W_q2)
        result["files_written"]["W_q2_attn_out_L0.npy"] = os.path.getsize(os.path.join(out_dir, "W_q2_attn_out_L0.npy"))
    
    result["W_ref_shape"] = W_ref.shape
    result["W_ref_bytes_F32"] = W_ref.nbytes
    result["W_ref_bits_per_weight"] = 32.0  # F32 reference
    result["classification"] = "EXTRACTED" if result_q2["status"] == "extracted" else "PARTIAL_EXTRACTION"
    result["metadata"] = {
        "note": "Q4 = IQ4_NL (~4-bit), Q2 = IQ2_XXS (~2-bit) in GGUF",
        "extracted_from_Q4_GGUF": os.path.basename(args.gguf_q4),
        "note_q4_tensor_type": f"type={result_q4['tensor_type']}",
    }
    
    print(json.dumps(result, indent=2, default=str))
    print(f"\nFiles written to {out_dir}/:")
    for f in result.get("files_written", {}):
        print(f"  {f}")


if __name__ == "__main__":
    main()
