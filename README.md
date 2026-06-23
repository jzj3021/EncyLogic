# EncyLogic

**RL- and MCTS-guided logic-synthesis optimization.** EncyLogic trains a Transformer
("SeqFormer") to generate sequences of Berkeley ABC synthesis commands that minimize a
circuit's delay/area, then uses Monte-Carlo Tree Search at inference time with a physical
(OpenROAD) reward to find high-quality command sequences for And-Inverter-Graph (AIG) circuits.

This repository is the **core method**, with all machine-specific paths factored into a single
config module (`encylogic/config.py`, overridable by environment variables). It does **not**
ship trained weights, benchmark circuits, or technology libraries — point the config at your own.

## Pipeline

The numbered stages form an end-to-end flow (`encylogic/pipeline/`):

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `step1_verilog_to_blif.py` | Verilog → BLIF via **Yosys** (parallel). |
| 2 | `step2_blif_to_aig.py` | BLIF → AIG via **ABC**, generating randomized structural variants (GIA space). |
| 3 | `step3_generate_sequences.py` | Run many random ABC command sequences per circuit; keep the best (nodes×levels) → training data. |
| 4 | `step4_jsonl_to_json.py` | Convert `.jsonl` → `.json`. |
| 5 | `step5_merge_jsonl.py` | Merge/validate multiple `.jsonl` datasets. |
| 6 | `step6_train_seqformer.py` | **Supervised** pre-training of the SeqFormer on the best sequences. |
| 7 | `step7_train_grpo.py` | **RL (Phys-GRPO)**: reward = real physical delay from the OpenROAD service. |
| 8 | `step8_inference.py` | Greedy step-wise inference (try top-k actions, keep the best by ADP). |
| 9 | `step9_mcts.py` / `step9_mcts_physical.py` | MCTS search — local ADP reward, or physical (OpenROAD) reward. |
| 10 | `step10_evaluate.py` | Final metrics: AND-nodes, levels, FPGA LUTs, ASIC area/delay (ABC `map`). |

Supporting: `encylogic/actionspace.py` (the ABC GIA action vocabulary), `configs/abc.rc`
(ABC aliases), `encylogic/pipeline/filter_best_sequences.py` (dataset refinement).

Physical reward backend (`encylogic/serving/`):
- `openroad_server.py` — a FastAPI service: takes `{design, sequence}`, applies the sequence in
  ABC, runs the OpenROAD ASAP7 flow, and returns delay/area/WNS/TNS.
- `client.py` — batch client that queries the service.

## Install

```bash
pip install -r requirements.txt
```

External tools (installed separately, referenced via env vars): **Berkeley ABC**, **Yosys**,
and (for physical evaluation) **OpenROAD / OpenROAD-flow-scripts** with an ASAP7 platform.

## Configure

All paths/binaries/endpoints live in `encylogic/config.py` and are overridable via environment
variables — copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
# edit ENCYLOGIC_ROOT, ABC_BIN, ENCYLOGIC_LIB, ENCYLOGIC_API_URL, ...
source .env
python -c "import encylogic.config as c; c.ensure_dirs(); print('workspace at', c.ROOT)"
```

Put your input AIG circuits in `$ENCYLOGIC_ROOT/benchmarks/`, your liberty at `ENCYLOGIC_LIB`,
and (optionally) trained checkpoints in `$ENCYLOGIC_ROOT/models/`.

## Run

```bash
# 0. (for physical reward) start the OpenROAD eval service
uvicorn encylogic.serving.openroad_server:app --host 0.0.0.0 --port 8088

# 1-5. build a training dataset
python -m encylogic.pipeline.step1_verilog_to_blif
python -m encylogic.pipeline.step2_blif_to_aig
python -m encylogic.pipeline.step3_generate_sequences
python -m encylogic.pipeline.step4_jsonl_to_json
# 6. supervised pre-training
python -m encylogic.pipeline.step6_train_seqformer
# 7. RL fine-tuning (needs the eval service)
python -m encylogic.pipeline.step7_train_grpo
# 9. MCTS search on a target circuit
python -m encylogic.pipeline.step9_mcts adder           # local ADP reward
python -m encylogic.pipeline.step9_mcts_physical adder  # physical reward (needs service)
# 10. evaluate optimized AIGs
python -m encylogic.pipeline.step10_evaluate
```

## Notes

- The two SeqFormer variants in the pipeline differ by design (a 256-dim decoder for SFT/greedy
  inference; a 512-dim encoder-decoder for RL/MCTS) — each script keeps its own definition.
- The method operates on **combinational** AIGs; the action vocabulary is ABC GIA (`&`) commands.
- This is research code released for reproducibility; interfaces are CLI/script-oriented.

## License

MIT — see `LICENSE`.
