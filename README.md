# HappyBot 
----- 


This repository aims to provide a light version of the HappyBot project - https://github.com/jdlamstein/happybot

 We focus on extracting only the Core from the solution, the detection structure and a new organization for the directories
 
 ## Preparing your env
 
 ```
$ virtualenv .venv --python=python3.6
$ source .venv/bin/activate
``` 

 ### Installing dependencies
 
 ```
 $ pip install -r requirements.txt
 ```
 
 ## How to use
 
Running for a specifc image:

```
python3 main.py --image=data/tests/img1.jpg
```
 

Running for a remote video:

```
 python3 main.py --video=http://192.168.0.3:8080/video
```

Running using your WebCam:

```
 python3 main.py
```