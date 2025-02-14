To reproduce the results in the report, please run the script: 

`python main.py -n [modelname] -r 10`
or the long-form script: `python main.py --model [modelname] --numruns 10`

Parameters: 
  [`--model`, `cnn1d`, `lstm`, `cnnlstm`, `transformer`, `PatchTST`]
  [`--numruns`, e.g., `10` for `10` runs]

If you want to retrain the model, you can add the parameter: `--retrain`
