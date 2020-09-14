from glob import glob 
import os, random, shutil

#Checks if test and train folders are there if not creates them
if ["val","test"] not in os.listdir():
    try:
        os.mkdir("val")
    except FileExistsError:
        pass
    try: 
        os.mkdir("test")
    except FileExistsError:
        pass

#Gets the class list
classes_list=glob("train/*")

#Loops on all classes
for classes_path in classes_list:

    #Val and test folder paths
    val_move_path = os.path.join("val",classes_path.split("/")[-1])
    test_move_path = os.path.join("test",classes_path.split("/")[-1])

    #Checks if class folder are there in val and test folders 
    if not os.path.isdir(val_move_path):
        os.mkdir(val_move_path)
    if not os.path.isdir(test_move_path):
        os.mkdir(test_move_path)

    #List of images in train/class 
    img_list=glob(os.path.join(classes_path,"*"))

    #Specify the percent you want to split
    percent=10
    img_to_move=int((percent/100)*len(img_list))

    #Moves images for val 
    for _ in range(img_to_move):
        move_img=random.choice(img_list)
        shutil.move(move_img,val_move_path)
        img_idx=img_list.index(move_img)
        del img_list[img_idx]

    #Moves images for test
    for _ in range(img_to_move):
        move_img=random.choice(img_list)
        shutil.move(move_img,test_move_path)
        img_idx=img_list.index(move_img)
        del img_list[img_idx]

    print("done for "+classes_path.split("/")[-1])
