# ObjectDetection_Proj
My company unofficial repository


## Resnet data learning for object detection
- general object detection script
```
python ssd_train_v1.py
```


# New! (23th Oct 2024)
# Manual control LCD dead pixel simulator

- all instruction of this tool is displayed on the screen
- Whenever you want to check, you can show and hide.
- In this code Japanese comment 

```
python lcd_tester.py
```


## Auto LCD inspection trainer
- appeared dead dot randamly, and inspecter should try 10 time challage
- add random dead size edtion

```
python int_screener.py
```
```
lcd_blkdot.py
```

```
python bigger_screener.py
```



- background black and show result Japanese version

```
python screener.py
```



## Real time OCR 
https://github.com/mi-kaneyon/ObjectDetection_Proj/tree/main/realtimeOCR

## pth_summary.py
- trained output which is pth file, checker.

```
python pth_summary.py /path/to/model.pth
```

## json_idadvchk.py
- coco format json file counter from json file

```
python json_idadvchk.py
```


## png_adjust.py
- Optimizing transparent

```
python png_adjust.py

```
## labeler.py
- Detected moving target

```
python labeler.py
```
  

## layercheck.py
- pth file layer difference check or statuschecker

```
python layercheck.py 
```

``` 
#Out puts ex ample Model's state_dict:
conv1.weight 	 torch.Size([64, 3, 7, 7])
bn1.weight 	 torch.Size([64])
bn1.bias 	 torch.Size([64])
bn1.running_mean 	 torch.Size([64])
bn1.running_var 	 torch.Size([64])
bn1.num_batches_tracked 	 torch.Size([])
layer1.0.conv1.weight 	 torch.Size([64, 64, 3, 3])


``` 

## image_autonum.py
- traing data padding tool named image_autonum.py
- You need modify base directory and target file names
- uplicates_per_image = 16000 should be changed by your target file numbers.
```
python image_autonum.py

```  

- Defective simulation (Proto type) is named "Dedective.ipynb"
Jupyter notebook ver.
Base of defective picture create test.

## crop_png.py
- Crop and optimize png format file for training data tool


```
python crop_png.py

```
- You can modifiy output/input folder before execute this script.

```
python resize json.py

```

## resize json.py
- Resize image file time support tool, which is rescale to bbox/segemantation coordinates

# MMdetection config files simple grammer(version matching checker)

## Projoct 
- My company management for AI thinking is very old school.
- Develop new project my self
- Open this space for the folks who try to create own way.
