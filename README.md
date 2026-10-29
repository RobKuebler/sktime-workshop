# sktime workshop (Sovereign AI Meetup Ulm, 2026-11-17)

Hands-on forecasting workshop, condensed from lectures 1, 2, 3 and 9 of
[sktime/forecasting-course](https://github.com/sktime/forecasting-course) (cloned in `forecasting-course/`).

| File | For |
|---|---|
| `sktime_workshop.ipynb` | participants: exercises with `...` stubs, no solutions |
| `sktime_workshop_instructor.ipynb` | you: solutions, instructor notes (tag `instructor`), executed outputs |
| `build/build_notebooks.py` | single source of truth, regenerates both notebooks |
| `build/make_illustrations.py` | renders the animated illustrations (8 animations and figures in `imgs/`, using the real data of the notebook) |
| `imgs/`, `data/` | images and the offline fallback for the Chronos demo; ship them next to the notebook |

## Participant setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab sktime_workshop.ipynb
```
The first code cells check the setup and download the Chronos model (about 35 MB) once.

## Edit and rebuild
Edit `build/build_notebooks.py`, then `python build/build_notebooks.py`.
To re-execute the instructor notebook:
`jupyter nbconvert --to notebook --execute --inplace sktime_workshop_instructor.ipynb`
