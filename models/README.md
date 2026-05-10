# Model Artifacts

This directory stores the pre-trained instructor baseline for BreastMNIST.

Files:
- `baseline_cnn.pt` — baseline CNN state dictionary (do not modify)
- `baseline_metrics.json` — baseline test-set accuracy reference (do not modify)

After running `python train.py` locally the following additional files appear
(they are ignored by Git):
- `student_cnn.pt` — your trained model weights
- `student_metrics.json` — your test accuracy and estimated score

Size policy: each model artifact must remain under 10 MB.
