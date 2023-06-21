# CGTnet


Requirements

To install requirements:

           pip install -r requirements.txt

To install CLIP

```
	   cd ./CLIP-main
	   python setup.py install
```



Datasets split:
           

           cd filelists/dataset
           python write_dataset_filelist.py


​           


Pretraining

To train the feature extractors in the paper, run this command:

            python pretrain.py       

Evaluation

To evaluate my model on NRT-Cls, run:

            python test.py



Hyperparameter setting

common setting:

           1-shot: k=10 kappa=9 beta=0.5 5-shot: k=4 kappa=1 beta=0.75

cross-domain setting1 :

           1-shot: k=10 kappa=1 beta=0.7 5-shot: k=4 kappa=1 beta=0.6

cross-domain setting2 :

           1-shot: k=10 kappa=1 beta=0.7 5-shot: k=4 kappa=1 beta=0.5

Contact the author e-mail：1134268255@qq.com
