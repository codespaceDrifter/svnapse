euclidean distance: how close two high dimensional points are

> $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

k-nearest neighbor (KNN):  
a classification or regression is made by calculating distances of new point from every point in the dataset, picking the k (odd number) examples with smallest distances, and averaging their target value (if classification pick the most seen type if regression calculate average value)

knn is a non parametric method, it does not use a training data

k too small: influenced by noise. k too big: influenced by target distribution among the population  

knn pros: simple, no training. knn cons: memory and compute intensive, sensitive to noise, cannot find deep patterns. 

KNNs can be used to do imputations based on the columns where datas are present to find k closest points then finding the average missing column values for them and imputing with that average