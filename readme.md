# Implementing LipNet


## Original Paper: 
https://doi.org/10.48550/arXiv.1611.01599


## Install dependencies and libraries


Create a new virtual conda environment to run this project. Installing on main environment may cause dependency issues.
```
conda create --name new_env python=3.80
conda install --file requirements.txt
conda activate new_env
```



## Train Model

Go to the directory contatining 'train.py' file and run the following command to train the model.
```
python train.py
```


## Test Model

Go to the Testing directory and write the following code to execute the testing file.
```
python test.py
```




## Open as Website

Go to the folder containing 'app.py' and run following command.
```
python app.py
```

This will open the website on the localhost:5000.
