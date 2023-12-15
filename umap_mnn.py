#writing cell content as .py file for simple import

# class UMAPMutualKNN(UMAP) definition
import numpy as np
import scipy
import heapq
import copy
from umap import UMAP
from umap.umap_ import find_ab_params, nearest_neighbors, fuzzy_simplicial_set, simplicial_set_embedding, dist
from sklearn.utils import check_random_state, check_array
from sklearn.cluster import KMeans
from sklearn import metrics
from joblib import hash

class UMAPMutualKNN(UMAP):
    # A method for greater cluster separation utilizing UMAP with a mutual nearest neighbor graph
    # Described in the paper:
    # @article{Dalmia2021UMAPConnectivity,
    #   author={Ayush Dalmia and Suzanna Sia},
    #   title={Clustering with {UMAP:} Why and How Connectivity Matters},
    #   journal={CoRR},
    #   volume={abs/2108.05525},
    #   year={2021},
    #   url={https://arxiv.org/abs/2108.05525},
    #   eprinttype={arXiv},
    #   eprint={2108.05525},
    #   timestamp={Wed, 18 Aug 2021 19:45:42 +0200},
    #   biburl={https://dblp.org/rec/journals/corr/abs-2108-05525.bib},
    #   bibsource={dblp computer science bibliography, https://dblp.org}
    #   }
    # and based on the implementation provided by the UMAP team in their documentation:
    # "Improving the Separation Between Similar Classes Using a Mutual k-NN Graph"
    # URL: https://umap-learn.readthedocs.io/en/latest/mutual_nn_umap.html
    # and the method github following the path nearest neighbors notebook:
    # URL: https://github.com/adalmia96/umap-mnn
    def __init__(self, random_state=None, spread = 1.0, learning_rate = 1.0, n_neighbors=30, new_n_neighbors=30, min_dist=0.1, gamma=1.0,  negative_sample_rate = 5, n_epochs = -1, connectivity='nearest', metric = 'jaccard', densmap = False, output_dens = False, parallel= False, verbose=False):
        super().__init__(n_neighbors=n_neighbors, verbose=verbose)
        #removed *args and *kwargs from __init__() and super().__init__()
        self.spread = spread
        self.learning_rate = learning_rate
        self.initial_alpha = self.learning_rate
        self.new_n_neighbors = new_n_neighbors
        self.min_dist = min_dist
        self.connectivity = connectivity
        self.verbose = verbose
        self.metric = metric  
        self._a, self._b = find_ab_params(self.spread, self.min_dist)
        self.gamma = gamma
        self.negative_sample_rate = negative_sample_rate
        self.n_epochs = n_epochs
        self.random_state = check_random_state(random_state)
        #random_state=self.random_state
        #metric=self.metric
        self.densmap = densmap
        self.output_dens = output_dens
        self.parallel = parallel
    
    def _min_spanning_tree(self, knn_indices, knn_dists, threshold):
      rows = np.zeros(knn_indices.shape[0] * self.n_neighbors, dtype=np.int32)
      cols = np.zeros(knn_indices.shape[0] * self.n_neighbors, dtype=np.int32)
      vals = np.zeros(knn_indices.shape[0] * self.n_neighbors, dtype=np.float32)
      
      pos = 0
      for i, indices in enumerate(knn_indices):
        for j, index in enumerate(indices[:threshold]):
          if index == -1:
            continue
          rows[pos] = i 
          cols[pos] = index
          vals[pos] = knn_dists[i][j]
          pos += 1
      
      matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
      Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(matrix)
      
      Tcsr = scipy.sparse.coo_matrix(Tcsr)
      weights_tuples = zip(Tcsr.row, Tcsr.col, Tcsr.data)
      
    
      sorted_weights_tuples = sorted(weights_tuples, key=lambda tup: tup[2])
      return sorted_weights_tuples

    def _create_connected_graph(self, mutual_nn, total_mutual_nn, knn_indices, knn_dists):
      connected_mnn = copy.deepcopy(mutual_nn)
      
      if self.connectivity == "nearest":
        for i in range(len(knn_indices)): 
          if len(mutual_nn[i]) == 0:
            first_nn = knn_indices[i][1]
            if first_nn != -1:
              connected_mnn[i].add(first_nn) 
              connected_mnn[first_nn].add(i) 
              total_mutual_nn += 1
        return connected_mnn
          
      #Create graph for mutual NN
      rows = np.zeros(total_mutual_nn, dtype=np.int32)
      cols = np.zeros(total_mutual_nn, dtype=np.int32)
      vals = np.zeros(total_mutual_nn, dtype=np.float32)
      pos = 0
      for i in connected_mnn:
        for j in connected_mnn[i]:
          rows[pos] = i 
          cols[pos] = j
          vals[pos] = 1
          pos += 1
      self.graph_ = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
      
      #Find number of connected components
      self.n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=self.graph_, directed=True, return_labels=True, connection= 'strong')
      print("n_components: {}".format(self.n_components))
      label_mapping = {i:[] for i in range(self.n_components)}
    
      for index, component in enumerate(labels):
        label_mapping[component].append(index)
    
      #Find the min spanning tree with KNN
      sorted_weights_tuples = self._min_spanning_tree(knn_indices, knn_dists, self.n_neighbors)
      
      #Add edges until graph is connected
      for pos,(i,j,v) in enumerate(sorted_weights_tuples):
    
        if self.connectivity == "full_tree":
          connected_mnn[i].add(j)
          connected_mnn[j].add(i) 
          
        elif self.connectivity == "min_tree" and labels[i] != labels[j]:
          if len(label_mapping[labels[i]]) < len(label_mapping[labels[j]]):
            i, j = j, i
            
          connected_mnn[i].add(j)
          connected_mnn[j].add(i)
          j_pos = label_mapping[labels[j]]
          labels[j_pos] = labels[i]
          label_mapping[labels[i]].extend(j_pos)
    
      return connected_mnn

    def _find_new_nn(self, knn_indices, knn_dists, knn_indices_pos, connected_mnn):
      new_knn_dists= [] 
      new_knn_indices = []
      
      for i in range(len(knn_indices)): 
        min_distances = []
        min_indices = []
        
        heap = [(0,i)]
        mapping = {}
              
        seen = set()
        heapq.heapify(heap) 
        while(len(min_distances) < self.n_neighbors and len(heap) >0):
          dist, nn = heapq.heappop(heap)
          if nn == -1:
            continue
        
          if nn not in seen:
            min_distances.append(dist)
            min_indices.append(nn)
            seen.add(nn)
            neighbor = connected_mnn[nn]
            
            for nn_nn in neighbor:
              if nn_nn not in seen:
                distance = 0
                if nn_nn in knn_indices_pos[nn]:
                  pos = knn_indices_pos[nn][nn_nn]
                  distance = knn_dists[nn][pos] 
                else:
                  pos = knn_indices_pos[nn_nn][nn]
                  distance = knn_dists[nn_nn][pos] 
                distance += dist
                if nn_nn not in mapping:
                  mapping[nn_nn] = distance
                  heapq.heappush(heap, (distance, nn_nn))
                elif mapping[nn_nn] > distance:
                  mapping[nn_nn] = distance
                  heapq.heappush(heap, (distance, nn_nn))
        
        if len(min_distances) < self.new_n_neighbors:
          for i in range(self.new_n_neighbors-len(min_distances)):
            min_indices.append(-1)
            min_distances.append(np.inf)
        
        new_knn_dists.append(min_distances)
        new_knn_indices.append(min_indices)
        
        if self.verbose and i % int(len(knn_dists) / 10) == 0:
            print("\tcompleted ", i, " / ", len(knn_dists), "epochs")
        if self.verbose and (i+1) == len(knn_indices):
            print("\tcompleted ", len(knn_indices), " / ", len(knn_dists), "epochs")
      return new_knn_dists, new_knn_indices

    def _mutual_nn_nearest(self, knn_indices, knn_dists, connectivity="min_tree"):
      mutual_nn = {}
      nearest_n= {}
    
      knn_indices_pos = [None] * len(knn_indices)
    
      total = 0
      
      for i, top_vals in enumerate(knn_indices):
        nearest_n[i] = set(top_vals)
        knn_indices_pos[i] = {}
        for pos, nn in enumerate(top_vals):
          knn_indices_pos[i][nn] = pos
      
      total_mutual_nn = 0
      for i, top_vals in enumerate(knn_indices):
        mutual_nn[i] = set()
        for ind, nn in enumerate(top_vals):
          if nn != -1 and (i in nearest_n[nn] and i != nn):
            mutual_nn[i].add(nn)
            total_mutual_nn += 1
    
      connected_mnn = self._create_connected_graph(
            mutual_nn, 
            total_mutual_nn, 
            knn_indices, 
            knn_dists
        )
      new_knn_dists, new_knn_indices = self._find_new_nn(
            knn_indices, 
            knn_dists, 
            knn_indices_pos, 
            connected_mnn
        )
      return connected_mnn, mutual_nn, np.array(new_knn_indices), np.array(new_knn_dists)
    
    def fit(self, X, y=None, force_all_finite=True):
        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C", force_all_finite=force_all_finite)
        self._raw_data = X
        self._input_hash = hash(self._raw_data)
        
        # Override the fit method to include the mutual k-NN graph creation
        random_state = check_random_state(self.random_state)

        # Find the original nearest neighbors
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            metric_kwds={},
            angular=False,
            random_state=self.random_state,
            low_memory=self.low_memory,
            use_pynndescent=True,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        
        # Create the connected mutual nearest neighbors graph
        _, _, self._knn_indices, self._knn_dists = self._mutual_nn_nearest(knn_indices, knn_dists)
        
        # Build fuzzy simplicial set
        if self.verbose:
            print("Begining Fuzzy Simplicial Set...")
        self.graph_, self.sigmas_, self.rhos_ = fuzzy_simplicial_set(
            X,
            self.new_n_neighbors,
            random_state=self.random_state,
            knn_indices=self._knn_indices,
            knn_dists=self._knn_dists,
            metric=self.metric,
            verbose = self.verbose
        )
        
        if self.verbose:
            print("Begining Simplicial Set Embedding...")
        # Embed the data
        if self.verbose:
            tqdm_dict = {'disabled': False}
        else:
            tqdm_dict = None
        self.embedding_, self._aux_data = simplicial_set_embedding(
            X,
            self.graph_,
            self.n_components,
            self.initial_alpha,
            self._a,
            self._b,
            self.gamma,
            self.negative_sample_rate,
            self.n_epochs,
            init=self.init,
            random_state=self.random_state,
            metric=self.metric,
            metric_kwds={},
            densmap=self.densmap,
            densmap_kwds={},
            output_dens=self.output_dens,
            output_metric=dist.named_distances_with_gradients["euclidean"],
            output_metric_kwds={},
            euclidean_output=True,
            parallel=self.parallel,
            verbose=self.verbose,
            #tqdm_kwds = tqdm_dict #in docs, but not accepted as param
        )
        self.embedding_ = np.nan_to_num(self.embedding_)
        return self
