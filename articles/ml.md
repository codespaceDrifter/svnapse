upervised learning: data includes desired outputs  
i.e. linear regression, decision tree, random forest, KNN, logistic regression  
unsupervised learning: data does not include desired outputs  

split data into train and test set. use train set to make the model learn. use test set to evaluate model. 

single linear regression:  
supervised. continuous target.  
feature selection with pearson's correlation coefficient  
>$$r_{xy} = \frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}$$

-1 $\le$ r $\ge$ 1  
positive correlation means as feature increases target increases, negative means feature increases target decreases. the greater the absolute value of r the higher the correlation and more valueble that feature is. absolute value wise, 0-0.3: weak, 0.3-0.7: moderate, 0.7-1.0: strong  

linear regression formula  
the model as f, $(x^{(i)}, y^{(i)}) = i^{th}$ training example
>$$\hat{y} = f(\vec{x}) = \theta_1x_1 + ... +\theta_nx_n+b$$

SSE loss formula
>$$J(\vec{\theta}) = \sum_i(\hat{y}-y)^2$$

gradient formula (just using $\theta_1$ as weight and $\theta_2$ as bias here):  

>$$\frac{\partial{J}}{\partial{\theta_1}} = 2\sum_i (\theta_1x_i+\theta_2-y)x_i$$

>$$\frac{\partial{J}}{\partial{\theta_2}} = 2\sum_i (\theta_1x_i+\theta_2-y)$$

gradient descent formula:
>$$\theta_{j next} =  \theta_j - \alpha \frac{\partial{J}}{\partial{\theta_j}}$$

reasoning: 
using taylor expansion:
$J(\theta + \Delta\theta) \approx J(\theta) + \frac{\partial{J}}{\partial{\theta}}\Delta\theta$  
(we discard the later terms in the taylor expansion due to the value being small)  
to make $J(\theta + \Delta\theta) < J(\theta)$ we set $\Delta\theta = -\alpha\frac{\partial{J}}{\partial{\theta}}$

terminology:
each $x^{(i)}$ is an n dimensional vector, and there are m such vectors total. $x^{(i)}$ denotes a row $x_i$ denotes a column

Logistic classification:
Used for binary classification

>$$\sigma (h_{\theta}(x)) = \frac{1}{1+e^{-h_{\theta}(x)}}$$

>$$\hat{y} = 1 \text{if } \sigma (h_{\theta}(x)) \ge 0.5 , 0 \text{otherwise}$$


unsupervised:  
determine structures in data. example: clustering. 