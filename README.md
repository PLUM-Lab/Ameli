>📋  Ameli: Enhancing Multimodal Entity Linking with Fine-Grained Attributes

# Ameli: Enhancing Multimodal Entity Linking with Fine-Grained Attributes

This repository is the official implementation of [Ameli: Enhancing Multimodal Entity Linking with Fine-Grained Attributes](https://aclanthology.org/2024.eacl-long.172). 

Please use the following citation:
```
@inproceedings{yao-etal-2024-ameli,
    title = "Ameli: Enhancing Multimodal Entity Linking with Fine-Grained Attributes",
    author = "Yao, Barry  and
      Wang, Sijia  and
      Chen, Yu  and
      Wang, Qifan  and
      Liu, Minqian  and
      Xu, Zhiyang  and
      Yu, Licheng  and
      Huang, Lifu",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.172",
    pages = "2816--2834",
    abstract = "We propose attribute-aware multimodal entity linking, where the input consists of a mention described with a text paragraph and images, and the goal is to predict the corresponding target entity from a multimodal knowledge base (KB) where each entity is also accompanied by a text description, visual images, and a collection of attributes that present the meta-information of the entity in a structured format. To facilitate this research endeavor, we construct Ameli, encompassing a new multimodal entity linking benchmark dataset that contains 16,735 mentions described in text and associated with 30,472 images, and a multimodal knowledge base that covers 34,690 entities along with 177,873 entity images and 798,216 attributes. To establish baseline performance on Ameli, we experiment with several state-of-the-art architectures for multimodal entity linking and further propose a new approach that incorporates attributes of entities into disambiguation. Experimental results and extensive qualitative analysis demonstrate that extracting and understanding the attributes of mentions from their text descriptions and visual images play a vital role in multimodal entity linking. To the best of our knowledge, we are the first to integrate attributes in the multimodal entity linking task. The programs, model checkpoints, and the dataset are publicly available at https://github.com/VT-NLP/Ameli.",
}
```
<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:
 

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->


## Dataset - Ameli

You can download dataset here:

- [Ameli version 1](http://nlplab1.cs.vt.edu/~menglong/project/multimodal/entity_linking/ameli/dataset/latest/). 

- Dataset Format and structure are explained in Section K.3 Dataset Format in Appendix of our paper (https://aclanthology.org/2024.eacl-long.172).

## Pre-trained Models

 
## Evaluation

 
 
-----------------------------------------------------------------------------------------


## Training (Optional)

 
<!-- 
>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Dataset Build (Optional)

 

<!-- ## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.  -->




## Contributing

>📋  Our dataset is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The associated codes are licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
