# snn-with-memristors

Paper available here: [Text Classification in Memristor-based Spiking Neural Networks](https://arxiv.org/abs/2207.13729)

This is a Pytorch-based sentiment analysis task in [the IMDB movie reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) using an SNN with a statistic memristor model [here](https://arxiv.org/abs/1703.01167).

This work takes two approaches to obtaining a trained memristor-based SNN: 1) converting a trained ANN to the memristor-based SNN, or 2) training the memristor-based SNN directly.

<center>
<img src="overview.svg#2" width="100%"/>
<br>
<i>Two approaches to obtaining a trained memristor-based SNN</i><br/>
</center>
<p></p>

Here is the project hierarchy:
- `rram_array.py` includes the virtual memristor array with the memristor model.
- `inputs.py` downloads the dataset and word embedding matrix (here we use [GloVe](https://nlp.stanford.edu/projects/glove/)), and transforms them to the forms that can be used in this work.
- `anntosnn.py` implements the conversion from a train-ANN to a memristor-based SNN (approach 1).
- `snntraining.py` implements the direct training of a memristor-based SNN (approach 2).
- `train.py` trains the network by choosing one of the approaches mentioned above.

### How to run
Before running, please make sure you have installed `numpy`, `torch`, `torchdata`, `torchtext` and `argparse`.

Run `python train.py` in the command prompt if you want to convert an ANN to a memristor-based SNN.

Run `python train.py --directConversion=False --thres=56.75` if you want to train a memristor-based SNN.

Rum `python trian.py --xbar=False` to disable virtual memristor arrays.

Please refer to `train.py` file to see the details regarding other parameters.

### Citation
```
@misc{rram_text_classification,
  doi = {10.48550/ARXIV.2207.13729},
  url = {https://arxiv.org/abs/2207.13729},
  author = {Huang, Jinqi and Serb, Alex and Stathopoulos, Spyros and Prodromakis, Themis},
  keywords = {Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Text Classification in Memristor-based Spiking Neural Networks},
  note = {arXiv:2207.13729},
  year = {2022}
}
```

### References

I referred to these Github projects when I made this work:
* https://github.com/bentrevett/pytorch-sentiment-analysis
* https://github.com/YigitDemirag/srnn-pcm
