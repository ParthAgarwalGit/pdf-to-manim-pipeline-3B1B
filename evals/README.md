# Evaluation Workspace

`benchmarks/` contains immutable scene inputs created from completed V2 runs. Keep frozen test
benchmarks separate from future fine-tuning data and split related scenes by source document.

`runs/` contains generated code, previews, diagnostics, provider records, costs, dashboards, and
human ratings. It is gitignored.

Start with:

```powershell
pdf2manim eval build output\DOCUMENT\RUN_ID --benchmark evals\benchmarks\scene-v1
pdf2manim eval validate evals\benchmarks\scene-v1
pdf2manim eval plan evals\benchmarks\scene-v1 --model gemini
pdf2manim eval run evals\benchmarks\scene-v1 --model gemini --max-cost 1.00
pdf2manim eval review evals\runs\EVALUATION_ID
```
