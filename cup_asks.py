# The new config inherits a base config to highlight the necessary modification
_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

# Conditional flag for using pretrained model
use_pretrained_model = True  # Change this flag as needed

# Determine the path for the pretrained model based on the flag
pretrained_path = 'checkpoints/your_pretrained_model.pth' if use_pretrained_model else None

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained=pretrained_path,  # Use your pretrained model based on the flag
    roi_head=dict(
        bbox_head=dict(
            num_classes=3  # New number of classes
        )
    )
)

# Modify dataset related settings to match cup_train.py
data_root = 'data/cup/'  # New data root directory
dataset_type = 'MyDataset'  # Dataset type in MyDataset.py

# Update the classes in your custom dataset
data = dict(
    train=dict(
        type=dataset_type,
        classes=('Furyo', 'cup', 'Ibutsu')  # Update the classes here
    ),
    val=dict(
        type=dataset_type,
        classes=('Furyo', 'cup', 'Ibutsu')  # Update the classes here
    )
)

# Optionally, you can freeze some layers to speed up training and reduce overfitting
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None, type='OptimizerHook')
# Freeze layers (this is just an example; you might need to adjust the layer names)
for param in model.backbone.parameters():
    param.requires_grad = False
