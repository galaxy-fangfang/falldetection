1. Create a directory: `FallAllD_pkl/` and put the file in it: https://drive.google.com/file/d/1LzCrxNCVVHs9Q-FPVfJtws6-E8L61qfk/view?usp=drive_link
   
2. To reproduce the results in the report, please run the script: 

   `python main.py -n [modelname] -r [numruns]`, for example `python main.py -n PatchTST -r 10`

3. Parameters: 

     `-n`, model name, [`cnn1d`, `lstm`, `cnnlstm`, `transformer`, `PatchTST`]
  
     `-r`, number of runs, e.g., `10` for `10` runs

4. If you want to retrain the model, you can add the parameter: `--retrain`
