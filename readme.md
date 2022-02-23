<h1>Make predictions of ICMEs by using the Nguyen model</h1>
https://github.com/gautiernguyen/Automatic-detection-of-ICMEs-at-1-AU-a-deep-learning-approach

##Description
<p>This repository contains a python script (main.py) which should be executed in 
a virtual environment in order to make prediction of ICMEs associated with a given start and stop times.
<br>

## How this work ?
To do so you need to follow these steps:</p>
👏
<ul>
    <li>you should have python 3.6 already installed for example => C:\Users\...\Programs\Python\Python36</li>
    <li>pip install virtualenv</li>
    <li>python -m virtualenv -p="C:\Users\...\Programs\Python\Python36\python.exe" venvwithpyhon36
    </li>
    <li>activate the virtual environment = venv\Scripts\activate</li>
    <li>install the requirements = pip --no-cache-dir install -r path/icme/requirements.txt</li>
    <li>run the script with the following arguments a start time <strong>start</strong>, a stop time <strong>stop</strong> and a destination folder <strong>path</strong> =>
    <strong> python -m main start stop path</strong>
    </li>
</ul>

## Dataset
<p>The script will automatically download the dataset if it does not exist in the data/datasets folder but if you want to download it manually, it can be found here: https://www.kaggle.com/gautiernguyen/wind-data:
</p>


 👏
