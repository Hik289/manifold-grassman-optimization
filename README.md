<h1 align="center">GrassRoute</h1>

<p align="center"><strong>Online Gradient Routing on the Grassmann Manifold</strong></p>

<p align="center">
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg" /></a>
  <a href="requirements.txt"><img alt="Python 3" src="https://img.shields.io/badge/Python-3-3776AB.svg" /></a>
</p>

<p align="center">
  <a href="assets/grassroute_intuition.png"><img src="assets/grassroute_intuition.png" alt="How GrassRoute works: split the gradient, apply Adam in a small subspace, update the weights, then update the subspace and align the stored Adam statistics." width="100%" /></a>
</p>

## How GrassRoute works

GrassRoute stores Adam's running gradient statistics along a small set of learned directions, which form a subspace. It updates the weights using both the gradient inside that subspace and the part outside it. The subspace changes during training to follow the gradients.

Each optimizer step has three parts:

1. **Split the gradient.** Project the gradient onto the current subspace and keep the remaining part, called the residual.
2. **Update the weights.** Apply Adam to the projected gradient, map the update back to the full weight matrix, and add a scaled gradient step for the residual.
3. **Update the subspace.** Move the basis toward the current gradient directions. Keep its columns perpendicular and of unit length, then align Adam's stored statistics with the new basis.

For a gradient matrix $G_t\in\mathbb{R}^{m\times d}$ and a basis $U_t\in\mathbb{R}^{d\times r}$, the split is

$$
Z_t=G_tU_t,\qquad G_t^{\perp}=G_t-Z_tU_t^\top.
$$

Adam's two running statistics each have size $m\times r$, rather than $m\times d$. The basis and its momentum each have size $d\times r$. The [paper source](paper/main.tex) gives the full update, implementation details, and proofs with their assumptions.

## Quick start

```bash
git clone https://github.com/Hik289/manifold-grassman-optimization.git
cd manifold-grassman-optimization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/optimizer_step.py
python experiments/test_numerics.py
```

The [example](examples/optimizer_step.py) runs one optimizer step on a small linear layer with generated data. The [numerical checks](experiments/test_numerics.py) verify eight properties of the gradient split and subspace updates. Both run on a CPU and need no dataset download.

## Use the optimizer

The example shows how to assign matrix weights to GrassRoute and use standard AdamW updates for the remaining parameters. The main settings are:

| Setting | What it controls |
| --- | --- |
| `rank` | Number of directions in the learned subspace |
| `lr` | Learning rate for the Adam update in the subspace |
| `lr_perp` | Learning rate for the residual update; defaults to `lr` |
| `grassmann_lr` | Step size for updating the subspace basis |
| `routed` | Whether a parameter group uses GrassRoute; one-dimensional parameters use the fallback optimizer |
| `fallback` | Optimizer for parameters outside GrassRoute: `adamw` by default, or `sgd` with momentum |

The implementation is in [experiments/grassroute.py](experiments/grassroute.py). Experiment scripts set their own ranks and learning rates, so use the settings in those scripts when reproducing a result.

## Reproduce the experiments

Run all commands from the repository root. Install the extra packages for training and plotting:

```bash
python -m pip install -r requirements-experiments.txt
```

### Recreate the paper's loss plots

The repository includes the saved training histories needed for all four loss plots. No training or dataset download is needed to redraw them:

```bash
python revision/core_loss.py
python revision/new_tasks_report.py
```

The scripts write figures to `paper/figures/`. The second script also updates the tables for the six public tasks. Generated PDFs stay local and are ignored by Git; the repository stores PNG figures.

### Run a short training check

```bash
python revision/suite.py --smoke --tasks matreg_128 --methods GrassRoute
```

This runs two training steps on generated matrix regression data. To try the earlier training script with a small run:

```bash
python experiments/train.py --family matreg --baseline B5 --epochs 2 --out_dir experiments/runs
```

Here, `B5` selects GrassRoute.

### Train on the six public tasks

| Task | Dataset | Main test metric |
| --- | --- | --- |
| Image denoising | Optical Recognition of Handwritten Digits | Mean squared error over all pixels; lower is better |
| Image inpainting | Optical Recognition of Handwritten Digits | Mean squared error over missing pixels; lower is better |
| Spam classification from tabular features | Spambase | Area under the ROC curve (AUROC); higher is better |
| Income classification | Adult | AUROC; higher is better |
| SMS spam classification | SMS Spam Collection | Macro-F1; higher is better |
| News topic classification | 20 Newsgroups | Macro-F1; higher is better |

The suite compares GrassRoute with SGD, AdamW, a fixed subspace, a subspace refreshed by periodic SVD, and SubTrack++. Install the exact SubTrack++ version used by the scripts:

```bash
bash scripts/install_subtrack.sh
```

The script downloads the [upstream source](https://github.com/criticalml-uw/SubTrack) at a fixed commit. The experiment adapter lets it run on the gradient's device without changing its optimizer calculations.

To retrain, use a separate clone and move the included results aside once. The runner skips experiments that already have a saved summary.

```bash
mv revision/newtask_results revision/newtask_results_bundled
python revision/prepare_new_tasks.py --family all
python revision/new_tasks_suite.py --register
python revision/new_tasks_suite.py
python revision/new_tasks_report.py
```

Data preparation downloads the datasets to `revision/newtask_data/` and fits preprocessing on the training split only. `--register` saves the experiment settings and file hashes; the next command starts training. New results are saved to `revision/newtask_results/`.

For each task and method, the script trains nine candidate settings for 200 epochs each, selects a setting using the validation score, then trains a fresh model for 1,000 epochs. It evaluates that final model on the test set once. These runs use one seed. The six-task suite runs on a CPU.

For other task families, `revision/` scripts read datasets from `GR_DATA_ROOT`, while the earlier `experiments/` scripts use `GRASSROUTE_DATA_ROOT`. Both default to the repository's `data/` directory. GPU and graph experiments need the appropriate PyTorch and PyTorch Geometric packages.

## What the results show

| Experiment | Finding |
| --- | --- |
| Earlier Burgers physics-informed neural network (PINN) experiment | GrassRoute achieved lower training loss than AdamW and periodic SVD in this single-seed run. |
| Cora graph classification | GrassRoute improved accuracy in the first comparison. Runs with other initializations did not consistently repeat the gain. |
| [SMS spam classification](revision/newtask_results/text_sms) | GrassRoute used 87.0% less optimizer-state memory than AdamW. Macro-F1 was 0.9463, compared with AdamW's 0.9548, in this single-seed run. |
| Full six-task comparison | All six methods completed all six tasks. GrassRoute did not beat both AdamW and periodic SVD on the main test metric in any task. |

The paper reports both favorable and unfavorable comparisons. The selected loss plots illustrate individual runs, not a win rate across tasks. The experiments use different settings; see the Experiments section and appendix for the details. These results do not establish statistical significance.

The repository includes the training histories used in the loss plots, their result summaries, and summaries of the validation search for the six public tasks. Raw datasets, model checkpoints, full training histories from the validation search, and machine logs are not included.

## Repository structure

```text
experiments/       optimizer, baselines, models, and original training scripts
revision/          later experiments, validation search, result summaries, and plots
eda/               early studies of gradient directions and candidate tasks
archive/           earlier code versions and plotting scripts
paper/             LaTeX source, bibliography and figure PNGs
assets/            method diagram and its drawing code
examples/          small example of using the optimizer
scripts/           setup script for the SubTrack++ baseline
```

Start with `examples/` to try the optimizer and `revision/` to run the later experiments. `archive/` preserves earlier versions. The `B6` baseline in the older code is a local approximation of SubTrack++; later experiments load the upstream implementation through the setup script above.

## Paper and citation

The [manuscript](paper/main.tex) is a working draft. Build it locally with pdfLaTeX and BibTeX:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

The draft is being prepared for an ICML 2027 submission and currently uses the included `icml2026` style. The citation below identifies the draft, not an accepted paper.

```bibtex
@misc{grassroute2027,
  title  = {GrassRoute: Online Gradient Routing on the Grassmann Manifold},
  author = {Anonymous Authors},
  year   = {2027},
  note   = {ICML 2027 working draft}
}
```

## License

The project code is available under the [MIT License](LICENSE). Files from other projects keep their original license notices.
