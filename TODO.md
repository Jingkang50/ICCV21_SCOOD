- Replace print with logging in `evaluator.py`
- Fix `test.py` (with reference to `train.py`)
- Combine `test.py` with `train.py`
- Add timm's WRN
    - What other models are used for 32x32 images?
- Compare optimization hyperparameters
- Schedule OE loss? Scheduling for clustering and pseudo-labeling?
    - Limit the pseudo-labels from contributing to the loss

- Try out resnet50
  - With reduced stride
  - With smaller feature dim?

- Archs to consider
  - WRN-40-4
  - WRN-40-2-0.3