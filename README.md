# ai-slop-detector

To set up the Python environment:

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n slop python=3.10 -y
conda activate slop
pip install -r requirements.txt
```

To install latex-related packages (linux)

```bash
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra cm-super dvipng fonts-liberation
```

## Download existing external evaluations

```bash
hf download ronanhansel/data-ai-slop-detector \
    --local-dir . \
    --repo-type dataset
```

## Using SOTA detector

Step 1: Select one user, filter all posts from that user with: at least 40 characters, no label.
Step 2: Select at most 20 random posts, perform SOTA detection sequentially.
Step 2.1: Fill in `ai_confidence` as a separate column.
Step 2.2: 10 posts are detected as human/ai, label all the remaining posts from the same user as human/ai.
Step 3: Select next user and perform from Step 1 to 2 again until no post left.
