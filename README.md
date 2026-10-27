# Scorpio

## Setup

Requires uv, rust, protoc and the CUDA 13 toolkit (also at runtime: SGLang
compiles kernels on the fly).

```bash
bash setup.sh && source .venv/bin/activate
curl -LO https://github.com/anirudhb11/scorpio/releases/download/v1.0/data.zip && unzip data.zip
```

## Run Scorpio

```bash
python -m scorpio.cli.prepare dataset=hmmt                             # sample and grade N=16 solutions per problem (hmmt | lcb | rg)
python -m scorpio.cli.post_process dataset=hmmt                        # what the judge reads: final answer (gpt-oss) or a summary (Qwen)
python -m scorpio.cli.score dataset=hmmt scorer=scorpio num_rounds=8   # judge random groups of K=4, over 8 partitions
python -m scorpio.cli.report dataset=hmmt scorer=scorpio               # accuracy, bias^2 and variance against cost
```

The default model is gpt-oss-20b; add `model=qwen3_5_4b` to every command for Qwen3.5-4B.

## Bias-variance plots

Score each scorer over at least two reps, then report and plot:

```bash
for s in pointwise v1 scorpio; do
  for rep in 0 1 2 3; do python -m scorpio.cli.score dataset=hmmt scorer=$s num_rounds=8 rep=$rep; done
  python -m scorpio.cli.report dataset=hmmt scorer=$s
done
python -m scorpio.cli.plot runs/*/metrics_*.json --out biasvar.pdf
```

## Add your own scorer

A scorer is one file in `scorpio/scorers/`. Each round it receives the problem
and all candidates, and returns one score per candidate (higher is better).
Call the model through `llm.chat`; rounds, seeding, resuming, transcripts and
token costs are handled for you.

```python
# scorpio/scorers/mine.py
import re

class Mine:
    def run_round(self, problem, candidates, scores_so_far, llm, rng, calls_so_far):
        scores = []
        for i, candidate in enumerate(candidates):
            prompt = (f"{problem}\n\nSolution:\n{candidate}\n\n"
                      "How likely is this solution to be correct? End with <score>1-10</score>.")
            reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=8192,
                             candidate_ids=[i]).texts[0]
            found = re.findall(r"<score>\s*(\d+)\s*</score>", reply)
            scores.append(float(found[-1]) if found else 5.0)
        return scores

def build(task, **params):
    return Mine()
```

Then run it like any other scorer, and it appears in `report` and `plot`:

```bash
python -m scorpio.cli.score dataset=hmmt scorer=mine num_rounds=8
```

`scorpio/scorers/pointwise.py` and `scorpio/scorers/scorpio.py` are fuller examples.

## Tree search

```bash
python -m scorpio.cli.search dataset=hmmt && python -m scorpio.cli.report_search dataset=hmmt
```

## SLURM

`slurm/run.sbatch <command> [overrides]` runs any command above as a one-GPU
job. To split a dataset over 4 GPUs:

```bash
for i in 0 1 2 3; do
  A="dataset=hmmt shard=$i/4"
  j=$(sbatch --parsable slurm/run.sbatch prepare $A)
  j=$(sbatch --parsable -d afterok:$j slurm/run.sbatch post_process $A)
  sbatch -d afterok:$j slurm/run.sbatch score $A scorer=scorpio num_rounds=8
done
```

Run `report` once every shard has finished. Add `--account`/`--partition` as
your cluster requires.

If compute nodes have no internet, download the model on a login node first
(`hf download openai/gpt-oss-20b`) and add `export HF_HUB_OFFLINE=1` to
`slurm/run.sbatch`.

## Citation

```bibtex
@article{buvanesh2026scorpio,
  title  = {On the Limits of Test-Time Compute for Self-Evaluation},
  author = {Buvanesh, Anirudh and Rouissi, Khaled and Yun, Sujin and Obando-Ceron, Johan and Moss, Morgane M and Lavoie, Samuel and Noukhovitch, Michael and Charlin, Laurent and Courville, Aaron},
  year   = {2026}
}
```

## Acknowledgement

* Inference runs on [SGLang](https://github.com/sgl-project/sglang).
* Parts of this code are borrowed from [V₁](https://github.com/HarmanDotpy/pairwise-self-verification).
