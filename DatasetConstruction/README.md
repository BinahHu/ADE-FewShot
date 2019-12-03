### Dataset Construction: Transform ADE20k to ADE-FewShot

This is an instruction for how to construct the most essential part of ADE-FewShot dataset.



> For now the framework of this part is not very elegant. Since some codes are missing, we simply offer some data files in json format at directory data/ and directory stat/. We will re-construct this part soon.



#### step1. Download ADE20k

```
bash download.sh
```

This will download and unzip the ADE20k dataset at the **parent directory** of the ADE-FewShot directory. If you want to save the dataset at other place, you may need to modify the directory parameter at the bash file and **other codes that refer to the origin ADE20k dataset**.

#### step2. Transform and split the dataset

```
bash run.sh
```

This will automatically detect the objects in each image of ADE20k and save their locations and annotations. Then it will split all the dataset into base and novel based on the occurrence of each category.  You can change the threshold of base and novel by modifying the parameter in split\_list.py

#### step3. Move relevant files into right position

```
bash move.sh
```

This will move all the data needed for the project to ../data/ADE/ADE\_Origin. The preprocess code will refer data files in this folder.



