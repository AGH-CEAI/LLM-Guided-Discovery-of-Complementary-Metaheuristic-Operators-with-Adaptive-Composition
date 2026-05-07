# LLM-Guided Discovery of Complementary Metaheuristic Operators with Adaptive Composition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GECCO 2026](https://img.shields.io/badge/GECCO-2026-red.svg)](https://gecco-2026.sigevo.org/)

> **LLM-based framework for automated discovery, specialization, and composition of metaheuristic operators into adaptive optimizers.**
> 
> Designed for broad task coverage via complementary operators and LLM-driven adaptive control. Evaluated on the **GNBG benchmark** for LLM-generated metaheuristics.

---

## Overview

This repository contains the implementation of the framework presented in our GECCO 2026 paper:

> **"LLM-Guided Discovery of Complementary Metaheuristic Operators with Adaptive Composition on the GNBG Benchmark"**

The key idea is to shift focus from selecting among predefined operators to **constructing operators automatically** using Large Language Models (LLMs). Rather than optimizing a single aggregated objective, our framework optimizes **task coverage** — producing complementary operators specialized for different problem characteristics, then composing them into an adaptive optimizer.

### Key Contributions

- **Automated Operator Discovery**: Uses LLMs (Claude Opus 4.6, MiniMax-2.7) to generate and refine metaheuristic operators from scratch
- **Task-Coverage Optimization**: Formulates operator specialization as `min max_t min_o e_{o,t}` to ensure every task is handled effectively by at least one operator
- **Adaptive Composition**: Integrates specialized operators via Thompson Sampling to dynamically select the most effective search behavior during optimization
- **State-of-the-Art Results**: Achieves global minima on **14 of 24** GNBG benchmark functions, with **9 functions solved consistently** across all 31 runs

---

## Methodology

The framework operates in four stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Seed Metaheuristic Generation                         │
│  ├── Generate N=4 candidate algorithms via Claude Opus 4.6      │
│  ├── Benchmark against previously discovered algorithms         │
│  └── Select best performer (geometric mean error) as base       │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Operator Selection                                    │
│  ├── Identify 3 critical operators from base algorithm          │
│  └── Selection based on structural importance & sensitivity     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Sequential Operator Specialization                    │
│  ├── Generate 10 variants per operator (MiniMax-2.7, 229B)      │
│  ├── Iterative refinement with per-task error feedback          │
│  └── Objective: min max_t min_o e_{o,t} (maximin coverage)      │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Adaptive Composition                                  │
│  ├── Synthesize 3 candidate adaptive algorithms (Claude Opus)   │
│  ├── Thompson Sampling for dynamic operator selection           │
│  └── Final optimizer combines complementary strengths           │
└─────────────────────────────────────────────────────────────────┘
```

### The Coverage Objective

Unlike traditional approaches that minimize average error, we formalize operator diversity as:

$$
\min \max_t \min_o e_{o,t}
$$

where $e_{o,t}$ is the error of operator $o$ on task $t$. This **maximin formulation** promotes diversity and specialization, ensuring each task is handled effectively by at least one operator in the portfolio.

---

## Repository Structure

```
.
├── README.md                           # This file
├── GNBG_Benchmark.py                   # Main benchmark runner
├── gnbg_loader.py                      # GNBG benchmark interface (24 functions)
├── gnbg_iii_loader.py                  # GNBG-III benchmark interface
├── evaluate_gnbg3.py                   # Evaluation script for GNBG-III
├── run_gnbg.py                         # Experiment orchestration
│
├── operator_generator.py               # LLM-based operator generation & refinement
├── adaptive_metaheuristics.py          # Adaptive composition mechanisms
├── continuous_adaptive_metaheuristics.py  # Continuous adaptive variants
├── best_algorithm.py                   # Final synthesized optimizer
│
├── replace_functions.py                # Utility for operator substitution
├── diagnose_hang.py                    # Debugging utility for runaway processes
│
├── run_adaptive_llama_cpp.ps1          # PowerShell runner (llama.cpp backend)
├── run_adaptive_mixed.ps1              # PowerShell runner (mixed backend)
│
├── llm_keys.json                       # LLM API configuration template
│
├── .vscode/                            # VS Code settings
├── GNBG-Python/                        # GNBG benchmark suite (source)
├── GNBG_III_Benchmarks_Python/         # GNBG-III benchmark suite (source)
└── runs/                               # Experimental results & logs
```

### Core Modules

| File | Description |
|------|-------------|
| `operator_generator.py` | Interfaces with LLMs (Claude, MiniMax) to generate, mutate, and refine operator code. Handles prompt engineering and feedback loops. |
| `adaptive_metaheuristics.py` | Implements Thompson Sampling and other selection mechanisms for dynamic operator composition during optimization. |
| `best_algorithm.py` | The final CMA-ES-based optimizer with adaptive sampling strategies (standard, diversity-increasing, archive-based). |
| `GNBG_Benchmark.py` | Standardized evaluation protocol: 31 independent runs, mean/std error, success rate (< 1e-8). |

---

## Results

Performance on the **GNBG suite (24 functions)** [1], averaged over **31 independent runs**:

| Function | Mean Error | Std Error | Success Rate |
|:--------:|:----------:|:---------:|:------------:|
| f1 | 8.665e-09 | 1.016e-09 | **31/31** |
| f2 | 5.032e-02 | 4.854e-04 | 0/31 |
| f3 | 8.508e-09 | 1.262e-09 | **31/31** |
| f4 | 8.492e-09 | 1.251e-09 | **31/31** |
| f5 | 9.752e-02 | 1.007e-03 | 0/31 |
| f6 | 9.771e-02 | 1.141e-03 | 0/31 |
| f7 | 8.745e-09 | 1.054e-09 | **31/31** |
| f8 | 8.744e-09 | 9.256e-10 | **31/31** |
| f9 | 2.573e+03 | 2.908e+03 | 0/31 |
| f10 | 8.720e-09 | 1.312e-09 | **31/31** |
| f11 | 8.561e-09 | 1.142e-09 | **31/31** |
| f12 | 8.635e-09 | 1.103e-09 | **31/31** |
| f13 | 2.676e+03 | 2.785e+03 | 0/31 |
| f14 | 2.965e+01 | 9.670e+01 | 1/31 |
| f15 | 6.646e+00 | 3.835e-01 | 0/31 |
| f16 | 4.511e+02 | 2.694e+02 | 8/31 |
| f17 | 5.378e+02 | 2.152e+02 | 4/31 |
| f18 | 5.325e+02 | 2.114e+02 | 4/31 |
| f19 | 4.360e+02 | 2.820e+02 | 9/31 |
| f20 | 4.167e-07 | 2.438e-08 | 0/31 |
| f21 | 5.000e+00 | 1.249e-11 | 0/31 |
| f22 | 5.000e+01 | 1.507e-10 | 0/31 |
| f23 | 9.485e-09 | 4.502e-10 | **31/31** |
| f24 | 1.079e+00 | 5.619e-01 | 0/31 |

**Summary:**
- ✅ **14/24** functions reached the global optimum (error < 1e-8)
- ✅ **9/24** functions solved consistently across all 31 runs (100% success rate)
- 🏆 High ranking potential based on previous competition results

---

## Installation

### Prerequisites

- Python 3.10+
- Git
- LLM API access (Claude Opus, MiniMax, or compatible endpoints)

### Setup

```bash
# Clone the repository
git clone https://github.com/AGH-CEAI/LLM-Guided-Discovery-of-Complementary-Metaheuristic-Operators-with-Adaptive-Composition.git
cd LLM-Guided-Discovery-of-Complementary-Metaheuristic-Operators-with-Adaptive-Composition

# Install dependencies
pip install -r requirements.txt  # if available, or install core packages:
pip install numpy scipy pandas matplotlib

# Configure LLM API keys
cp llm_keys.json.template llm_keys.json
# Edit llm_keys.json with your API credentials
```

### LLM Configuration

The framework supports multiple LLM backends:

- **Claude Opus 4.6** (Anthropic) — Used for high-level algorithm generation and adaptive composition
- **MiniMax-2.7 (229B)** — Used for efficient iterative operator refinement
- **llama.cpp** — Local inference support via PowerShell runners

Configure endpoints in `llm_keys.json`:

```json
{
  "anthropic": {
    "api_key": "your-anthropic-key",
    "model": "claude-opus-4-6-20261101"
  },
  "minimax": {
    "api_key": "your-minimax-key",
    "model": "MiniMax-M2.7"
  },
  "llama_cpp": {
    "endpoint": "http://localhost:8080/completion",
    "model_path": "path/to/gguf"
  }
}
```

---

## Usage

### Running the Full Framework

```bash
# Stage 1-4: Complete pipeline from seed generation to adaptive composition
python run_gnbg.py --mode full --runs 31 --functions all
```

### Running Individual Stages

```bash
# Stage 1: Generate seed metaheuristics
python operator_generator.py --stage seed --n_candidates 4

# Stage 3: Specialize operators (requires base algorithm from Stage 1)
python operator_generator.py --stage specialize --operator mutation --variants 10

# Stage 4: Compose adaptive optimizer
python adaptive_metaheuristics.py --compose --operators_dir ./runs/operators/
```

### Evaluating the Final Algorithm

```bash
# Benchmark against GNBG suite
python GNBG_Benchmark.py --algorithm best_algorithm.py --runs 31 --output results/

# Evaluate on GNBG-III
python evaluate_gnbg3.py --algorithm best_algorithm.py
```

### Using PowerShell Runners (Windows)

```powershell
# Run with llama.cpp backend
.\run_adaptive_llama_cpp.ps1 -Functions 1-24 -Runs 31

# Run with mixed backend (Claude + local)
.\run_adaptive_mixed.ps1 -Functions all -Runs 31
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{anonymous2026llm,
  title={LLM-Guided Discovery of Complementary Metaheuristic Operators with Adaptive Composition on the GNBG Benchmark},
  author={Anonymous Author(s)},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference 2026 (GECCO '26)},
  year={2026},
  organization={ACM},
  address={San Jos\'{e}, Costa Rica}
}
```

---

## References

[1] Amir H. Gandomi, Mohammad Nabi Omidvar, Rohit Salgotra, and Kalyanmoy Deb. 2025. *A Generalized and Configurable Benchmark Generator for Continuous Unconstrained Numerical Optimization.* arXiv:2312.07083 [cs.NE]

[2] Nikolaus Hansen and Andreas Ostermeier. 2001. Completely Derandomized Self-Adaptation in Evolution Strategies. *Evolutionary Computation* 9, 2, 159–195.

[3] MiniMax. 2026. MiniMax 2.7 Language Model. https://huggingface.co/MiniMaxAI/MiniMax-M2.7

[4] Niki van Stein and Thomas Bäck. 2025. LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics. *IEEE Transactions on Evolutionary Computation* 29, 2, 331–345.

[5] Rainer Storn and Kenneth Price. 1997. Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. *Journal of Global Optimization* 11, 4, 341–359.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Generative AI–based tools were utilized for language editing, text refinement, and as the main instrument for conducting the experiments.
- Benchmark suites provided by the GNBG project [1].
- Competition organized by the GECCO 2026 LLM-Generated Metaheuristics track.

---

## Contact

For questions or issues, please open an [Issue](https://github.com/AGH-CEAI/LLM-Guided-Discovery-of-Complementary-Metaheuristic-Operators-with-Adaptive-Composition/issues) or contact the AGH-CEAI research group.

---

<p align="center">
  <i>Built with 🤖 LLMs and 🧬 Evolutionary Computation</i>
</p>
