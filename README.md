# ENSEMBLE-techniques
Ensemble techniques are a type of machine learning approach that involves combining multiple models to improve their predictive power. Instead of relying on a single model to make predictions, ensemble methods create a group of models that work together to produce a more accurate prediction than any individual model could.

![ensemblepng](https://github.com/dame-cell/ensemble-techniques/assets/122996026/11be2b2b-3d18-43c1-a43a-2c0af6e6bd45)

## Types of ensemble learning :
![Blue and Purple Casual Corporate App Development Startup Mind Map Graph](https://github.com/dame-cell/ensemble-techniques/assets/122996026/d1d93320-568a-4a58-93a2-139ab04c0fcc)


## VOTING ENSEMBLE 
A voting ensemble, also known as a voting classifier or voting regressor, is an ensemble technique in machine learning where multiple models are combined to make a prediction or decision based on a majority vote or averaging
* After training a voting ensemble model, the output is a combined model that represents the aggregation of predictions from multiple individual models. This combined model is commonly referred to as the "ensemble model" or * "voting classifier/regressor," depending on the task (classification or regression).

* The ensemble model can be used to make predictions on new, unseen data by feeding the input to each individual model and aggregating their predictions according to the voting scheme. The specific output of the ensemble * model depends on whether it's a classification or regression task:

* Classification: The ensemble model outputs the majority class predicted by the individual models in the case of hard voting. Alternatively, in the case of soft voting, it can output the class with the highest average * predicted probability across the individual models.
* Regression: The ensemble model outputs the average or weighted average of the predicted values from the individual models. The weights can be assigned based on the performance or expertise of each model.

* The advantage of using a voting ensemble is that it can often improve the overall accuracy and robustness of predictions compared to individual models. It helps to leverage the diverse strengths of different models and mitigate the impact of potential weaknesses.

![Dark Green and Light Blue Modern Mind Map and Brainstorm Graph](https://github.com/dame-cell/ensemble-techniques/assets/122996026/d49decad-46fd-45aa-a690-0d14dd7da54e)

## here is the link to the code



## Stacking: 
* Stacking is a technique that involves combining the predictions of multiple models using a meta-model. The meta-model takes the predictions of the individual models as input and produces a final prediction.
* After training a model using stacking, the output is a meta-model or a stacked model that combines the predictions of multiple base models. The stacked model is designed to learn and make predictions based on the patterns and relationships observed in the predictions of the base models.

* Stacking is an ensemble technique that involves training multiple base models on the training data. The predictions of these base models are then used as features to train a meta-model, which learns to make predictions based on the collective insights of the base models.

* The specific output of the stacking ensemble depends on whether it's a classification or regression task:

* Classification: In classification tasks, the stacking ensemble combines the predictions of the base models to train a meta-model that produces the final predicted class labels. The base models predict the class labels for the training instances, and these predictions serve as features for the meta-model. The meta-model is typically a classifier, such as logistic regression, that takes the base models' predictions as input and learns to make the final class label predictions.

* Regression: In regression tasks, the stacking ensemble combines the predictions of the base models to train a meta-model that produces the final predicted values. Similar to classification, the base models predict the values for the training instances, and these predictions are used as features for the meta-model. The meta-model, often a regression model such as linear regression, takes the base models' predictions as input and learns to make the final regression predictions.

* The advantage of using stacking is that it allows the ensemble to capture the strengths of individual base models and create a more powerful and accurate meta-model. By leveraging the diversity of the base models, stacking can often improve the predictive performance and generalization of the ensemble.

![Dark Green and Light Blue Modern Mind Map and Brainstorm Graph (1)](https://github.com/dame-cell/ensemble-techniques/assets/122996026/a8f21032-e907-466b-846f-37d3860ec943)


## BAGGING:
* bagging, is a technique that involves creating multiple models using different random subsets of the training data. These models are then combined to make a final prediction.
After training a model using bagging, the output is an ensemble of multiple models, each trained on a different subset of the training data. The combined model is known as a "bagging ensemble" or a "bagging classifier/regressor."

* Bagging, short for Bootstrap Aggregating, is an ensemble technique that involves creating multiple models using bootstrapped samples of the training data. Each model is trained independently on its own bootstrap sample, which is a randomly selected subset of the original training data with replacement. The predictions of these models are then aggregated to make a final prediction.

* In the case of classification tasks, the bagging ensemble combines the predictions of the individual models to determine the final predicted class label. This can be done through voting, where the majority class predicted by the models is selected as the final prediction.

* In the case of regression tasks, the bagging ensemble combines the predicted values from the individual models to determine the final predicted value. This can be done through averaging or taking the median of the predicted values.

* The advantage of using bagging is that it helps to reduce variance and improve the overall stability and accuracy of the model. It leverages the diversity of multiple models trained on different subsets of the data, leading to more robust predictions. Bagging ensembles are commonly used with decision trees (random forests) or other base models to enhance their performance.
* repeat process as in find the average after training the model independently 

** SOME ALGORTHIMS FOR BAGGING 

* Random Forest: Random Forest is a bagging algorithm that combines multiple decision trees to make predictions. Each decision tree is trained on a random subset of the training data, and the final prediction is obtained by aggregating the predictions of individual trees, typically using majority voting for classification or averaging for regression. Random Forest also introduces randomness in the feature selection process,
*  
![Dark Green and Light Blue Modern Mind Map and Brainstorm Graph (3)](https://github.com/dame-cell/ensemble-techniques/assets/122996026/1dcfbb20-eb85-404b-a655-2027482960cc)


## Boosting:
* Boosting is a technique that involves creating a sequence of models that each attempt to correct the errors of the previous models. This approach is particularly effective when dealing with complex datasets.
* After training a model using boosting, the output is an ensemble model that combines multiple weak or base models to form a strong predictive model. This ensemble model is commonly referred to as a "boosting ensemble" or a "boosting classifier/regressor."

* Boosting is an ensemble technique that works by sequentially training models in a way that each subsequent model focuses on the samples that were misclassified or have high errors by the previous models. This iterative process aims to improve the overall performance of the ensemble by gradually reducing the bias and increasing the model's predictive power.

* The specific output of the boosting ensemble depends on whether it's a classification or regression task:

* Classification: The boosting ensemble in classification tasks combines the predictions of the individual models to determine the final predicted class label. Each base model, also called a weak classifier, assigns weights to the training instances based on their importance. In the end, the predictions of the weak classifiers are combined, typically using a weighted voting scheme, to produce the final prediction.

* Regression: In regression tasks, the boosting ensemble combines the predictions of the individual models to determine the final predicted value. Similar to classification, each weak model assigns weights to the training instances based on their importance. The predictions of the weak models are combined using a weighted averaging scheme, where the weights correspond to the importance of each model's prediction.

* The advantage of using boosting is that it focuses on correcting the errors made by previous models, leading to a highly accurate ensemble model. Boosting is effective at handling complex datasets and can often outperform individual models. Gradient Boosting and AdaBoost are popular boosting algorithms used in practice.

** SOME ALGORITHMS FOR BOOSTING

* AdaBoost (Adaptive Boosting): AdaBoost is one of the earliest and widely used boosting algorithms. It works by assigning weights to training instances and iteratively training weak models on these weighted instances. Each subsequent model focuses more on the misclassified instances from previous models, thereby improving the overall performance of the ensemble.

* Gradient Boosting: Gradient Boosting is a general framework for boosting that aims to minimize a loss function by adding models sequentially. The models are trained to predict the gradient of the loss function, and the predictions are combined to update the ensemble iteratively. Gradient Boosting can be used with various loss functions and is known for its flexibility and strong performance.

* XGBoost (Extreme Gradient Boosting): XGBoost is an optimized implementation of Gradient Boosting that utilizes parallel processing, tree pruning, and regularization techniques. It is known for its efficiency and scalability, making it a popular choice for many machine learning tasks. XGBoost has gained significant popularity in data science competitions due to its superior performance.

* LightGBM (Light Gradient Boosting Machine): LightGBM is another optimized implementation of Gradient Boosting that focuses on achieving faster training speed and lower memory usage. It utilizes histogram-based algorithms and features like leaf-wise tree growth and gradient-based one-sided sampling. LightGBM is often preferred for large-scale datasets and real-time applications.

* CatBoost: CatBoost is a boosting algorithm that is designed to handle categorical features effectively. It automatically handles categorical variables by using various encoding techniques and is robust to missing values. CatBoost also incorporates advanced regularization techniques and provides good performance out of the box.
* 
![Dark Green and Light Blue Modern Mind Map and Brainstorm Graph (4)](https://github.com/dame-cell/ensemble-techniques/assets/122996026/e6a05589-888a-416a-a245-8df3c72e31f6)


