@echo off
cd /d "%~dp0"
uv run --locked --extra neural python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto fail
uv run --locked --extra neural pytest -q --basetemp=.pytest-tmp || goto fail
uv run --locked --extra neural python scripts/run_qrs_benchmark.py || goto fail
uv run --locked --extra neural python scripts/run_classical_classification.py || goto fail
uv run --locked --extra neural python scripts/train_beat_classifier.py --skip-existing || goto fail
uv run --locked --extra neural python scripts/evaluate_beat_classifier.py || goto fail
echo.
echo Finished. Results are in results\qrs_detection, results\classical_classification and results\beat_classification.
pause
exit /b 0
:fail
echo.
echo A step failed. Read the message above.
pause
exit /b 1
