#writing cell content as .py file for simple import

# An autoencoder clustering method from the paper
# Unsupervised Deep Embedding for Clustering Analysis
# by Junyuan Xie, Ross Girshick, and Ali Farhadi
# https://arxiv.org/pdf/1511.06335.pdf
# and based on David Ko's example implementation of their method:
# https://ai-mrkogao.github.io/reinforcement%20learning/clusteringkeras/

from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf


from sklearn.cluster import KMeans

# David Ko's Custom Clustering layer
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform',
            name='clusters'
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
        
    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
# preparing an sklearn-like class for later ease of use, following the Deep Embedding Clustering description:
class DeepClusteringModel:
    # An autoencoder clustering method from the paper
    # Unsupervised Deep Embedding for Clustering Analysis
    # by Junyuan Xie, Ross Girshick, and Ali Farhadi
    # https://arxiv.org/pdf/1511.06335.pdf
    # and based on David Ko's example implementation of their method:
    # https://ai-mrkogao.github.io/reinforcement%20learning/clusteringkeras/
    def __init__(self, random_state, n_clusters=10, n_pca_components=256, alpha=1.0):
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self.encoder = None
        self.model = None
        self.alpha = alpha
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        
    def _create_autoencoder(self, dims, act='relu'):
        n_stacks = len(dims) - 1
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        for i in range(n_stacks-1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer='glorot_uniform', name='encoder_%d' % i)(x)
        encoded = Dense(dims[-1], kernel_initializer='glorot_uniform', name='encoder_%d' % (n_stacks - 1))(x)
        x = encoded
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer='glorot_uniform', name='decoder_%d' % i)(x)
        x = Dense(dims[0], kernel_initializer='glorot_uniform', name='decoder_0')(x)
        decoded = x
        autoencoder = Model(inputs=input_img, outputs=decoded, name='AE')
        encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
        return autoencoder, encoder
    
    def _target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def _get_callbacks(self, tol, verbose):
        early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=tol, verbose=verbose)
    
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
    
        lr_scheduler = LearningRateScheduler(scheduler)
    
        return [early_stopping, lr_scheduler]
    
    def fit(self, X, maxiter=10000, batch_size=256, tol=0.001, verbose=0):
        
        # Define the autoencoder architecture
        dims = [X.shape[-1]] + [500, 500, 2000, self.n_clusters]
        autoencoder, self.encoder = self._create_autoencoder(dims)
        
        # Initialize KMeans clustering and predict cluster centers
        y_pred = self.kmeans.fit_predict(self.encoder.predict(X))
        
        # Define and compile the clustering model
        clustering_layer = ClusteringLayer(self.n_clusters, alpha = self.alpha, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        self.model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        
        # Set initial cluster weights using KMeans centroids
        self.model.get_layer(name='clustering').set_weights([self.kmeans.cluster_centers_])

        callbacks = self._get_callbacks(tol, verbose)

        # Update target distribution, which is used as the 'label' in training
        q = self.model.predict(X, verbose=verbose)
        p = self._target_distribution(q)

        # fit the model
        self.model.fit(X, p, batch_size=batch_size, epochs=maxiter, callbacks=callbacks, verbose=verbose)
        
        return self
            
    def predict(self, X):
        # Use the encoder and k-means to predict the cluster labels
        encoded_X = self.encoder.predict(X)
        return self.kmeans.predict(encoded_X)
        
    def fit_predict(self, X, maxiter=10000, batch_size=256, tol=0.001, verbose=0):
        self.fit(X, maxiter, batch_size, tol, verbose)
        return self.predict(X)
