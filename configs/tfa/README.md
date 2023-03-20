## Naming:

new_branch_contrastive_loss_cls_region_separate, new_branch_contrastive_loss_cls_region_nobbox_separate are the base model for the contrastive RoI head and are used in the other config file.
They indicate that the Sub-Parts are used for classification and not regression if nobbox is in the name.

## Training setup files:

# Base training files:
The different subfolders are indicating the conducted base training before completing fine-tuning.

The other files in this folder are the setup files for base training.

Eventhough an exemple exists with the GIoU, no results are to be extracted and is only here as demonstration of the inclusion of a "contrastive" regression loss.

Otherwise the naming is as follow:

- bboxwX indicates that regression weight has been kept constant during base training. wbbox_high stands for a regression weight of 10 and 1 otherwise.
-bbox_changing_w/changingw indicates that regression weight has been varied during base training.

- if nobg is mentionned, it indicated that the model was trained excluding background proposals in the contrastive loss. Otherwise background proposals are kept.

- if percalass is mentionned, it means that the queue has a per class structure, otherwise a random one is adopted.

# Fine-tuning Files:

- TrueFT indicates a fine-tuning where X-novel shots are used and 1 base shot. EXCEPT if stated otherwise, i.e. 5base_10shots, which indicates 10 novel shots and 5 base ones.

- All naming convention before Xshots refeeres to the base training parameters. I.e. contrastive_base_1_perclass_cls_loss_10shots indicates a contrastive base training with a regression weight fixed to 1 and a queue per class

- contrastive_separate_nobbox states that the Sub-Parts have been used in the classification head but not in the regression one. The separate indicates that the classification loss has been computed separly for the Sub-Parts and the true object

- basePL indicates the "base pipeline", i.e. '_base_/datasets/testing_norm_optimize_multi_rotate_s2.py'

- if notlight is present it means that the supervised contrastive loss computes the similarities before removing the proposals with low IoU

- wX indicates that a regresion weight of X is used during

- nomean indicates that the queue is using the different samples and not a mean of them (it has been tested during dev. but not investigated more)

- no/with bg indicates the background inclusion/exclusion of the queue

- rnd presence in the name indicates a random queue architecture

- more2 is inherited for previous tests but is not of impotance anymore

## Other informations:
split2/browse_dataset*.py allows to browse the image in the dataset with the transformation appplied, except the norm in order to have optical images for illustration/bbox checks/etc..

# Other setup files

all 'new_branch_contrastive*.py' are preivous contrastive head base model used in base tests

tfa_r50_unfreeze*.py are the files used for baseline establishement