# ObjectDetection_Proj
My company unofficial repository
# New! (2nd Feb 2024)

## png_adjust.py
- Optimizing transparent

```
python png_adjust.py


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
