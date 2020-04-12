# eval_otb
A  python toolkit for evaluating your tracking result on datasets: 
[OTB100](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html), 
[LaSOT](https://cis.temple.edu/lasot/) and
[UAV123](https://uav123.org/)



## Usage

### Requirements

```
matplotlib
opencv-python
numpy
```



### Tracking results placement

Your tracking result should be placed on the ***Results*** folder, and you can name your result folder as the name of your tracker for convenience.

```shell
|――Results
	|――ATOM
		|――default_1
			|――Basketball.txt
			|――...
		|――default_2
		|――default_3
	|――DiMP18
		|――Basketball.txt
		|――...
	|――DiMP50
    	|――Basketball.txt
		|――...
	|―― ...

```

As you can see above, your tracking result may have several repeated attempts (as some trackers behave differently every time, for example ***ATOM*** ). The evaluator will output the average value for these kind of trackers.



### Code modification

To run this toolkit on your working space, you will have to do a little modification on the code. Let's take eval_otb for example:

- Find the following code in ***plot.py*** and replace OTB_path with your own path.

```python
###################################################
# Here to implement the path to your OTB100 dataset
###################################################
OTB_path = "/media/myy/Data/OTB100"
evalotb = EvalOTB(OTB_path)
```

- Find the following code in ***plot.py*** and implement your tracking result params. Remember if your tracking result has several repeated attempts, set the "repeat" value to True.

```python
##############################################################
# Here to implement your result params
# trackers[NAME] = {"path": PATH, "repeat": Bool}
#   -PATH: Path to your tracking result
#   -repeat: whether tracking result contains several repeats, 
#			 if True, output the average result
##############################################################
trackers = {}
trackers["ATOM"] = {"path": "/media/myy/Data/Repo/eval_otb/Results/ATOM", "repeat": True}
trackers["DiMP18"] = {"path": "/media/myy/Data/Repo/eval_otb/Results/DiMP18", "repeat": False}
trackers["DiMP50"] = {"path": "/media/myy/Data/Repo/eval_otb/Results/DiMP50", "repeat": False}
```



### Run

After all configurations done, you are able to run the following command to evaluate your tracking result:

```python
python plot.py
```

If setup correctly, the output will look like:

![](./.figs/result_otb.png)

<center><i>Result on OTB100</i></center>

![](./.figs/result_uav.png)

<center><i>Result on UAV123</i></center>

![](./.figs/result_lasot.png)

<center><i>Result on LaSOT</i></center>