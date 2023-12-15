# Interactive Image Classification on CIFAR-10
- Exploratory Analysis of CIFAR-10 Dataset
- Custom Metric Implementations for UMAP Dim Reduction on Image Datasets 
- Custom Model Implementation of UMAP_MNN Dim Reduction
~~~
      A method for greater cluster separation utilizing UMAP with a mutual nearest neighbor graph
      Described in the paper:
      @article{Dalmia2021UMAPConnectivity,
        author={Ayush Dalmia and Suzanna Sia},
        title={Clustering with {UMAP:} Why and How Connectivity Matters},
        journal={CoRR},
        volume={abs/2108.05525},
        year={2021},
        url={https://arxiv.org/abs/2108.05525},
        eprinttype={arXiv},
        eprint={2108.05525},
        timestamp={Wed, 18 Aug 2021 19:45:42 +0200},
        biburl={https://dblp.org/rec/journals/corr/abs-2108-05525.bib},
        bibsource={dblp computer science bibliography, https://dblp.org}
        }
      and based on the implementation provided by the UMAP team in their documentation:
      "Improving the Separation Between Similar Classes Using a Mutual k-NN Graph"
      URL: https://umap-learn.readthedocs.io/en/latest/mutual_nn_umap.html
      and the method github following the path nearest neighbors notebook:
      URL: https://github.com/adalmia96/umap-mnn
~~~
- Custom Model Implementation of Unsupervised Deep Embedding for Clustering Analysis
~~~
      An autoencoder clustering method from the paper
      Unsupervised Deep Embedding for Clustering Analysis
      by Junyuan Xie, Ross Girshick, and Ali Farhadi
      https://arxiv.org/pdf/1511.06335.pdf
      and based on David Ko's example implementation of their method:
      https://ai-mrkogao.github.io/reinforcement%20learning/clusteringkeras/
~~~
- Brief Model Performance Base Case Comparisons
- Image Augmentation Techniques
- Interactive Web-Application for Users to Try the Different Clustering Techniques + Dataset Augmentations
