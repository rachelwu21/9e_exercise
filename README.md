# ML Model Served via Web API

## Notes

The model script is in `utils.py` and `script.py`. 

Uncommenting `try_many_models()` in `script.py` outputs a file `scores.txt` that lists the score of each model tried and its (non-default) parameters. It also prints the best score (mean of the negative absolute error) of all the models tried.

The current `script.py` outputs the model I used as `final.model`.