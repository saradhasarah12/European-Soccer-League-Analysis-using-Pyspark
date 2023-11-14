# European-Scoccer-League-Analysis-using-Pyspark

***INTRODUCTION***
      The project involves utilizing big data analytics to analyze and derive insights from the vast ecosystem of the global sports and media industries. Key focus areas include on-field events analysis, team performance assessment with different player combinations, and tracking the historical performance of athletes or teams over seasons.

***DATASET***
**https://www.kaggle.com/datasets/secareanualin/football-events**

   The dataset available at the provided link on Kaggle, titled "Football Events," is a comprehensive collection of football-related data capturing a wide array of events occurring during matches. It encompasses details such as player actions, team statistics, match information, and contextual attributes during football events. With a focus on European soccer, the dataset includes features like player and team identifiers, match outcomes, event types (such as passes, shots, and fouls), player positions, and more. This rich and detailed dataset offers a valuable resource for researchers, analysts, and enthusiasts seeking to explore and analyze the intricacies of football matches, players, and teams. The diverse nature of the data makes it suitable for a range of analyses, from player performance assessments to strategic insights and trend identification within the dynamic realm of football.

***PYSPARK MACHINE LEARNING ALGORITHMS***
1. **Naive Bayes (NaiveBayes):**
   - **Usage in the Code:** It's used for classification (`NaiveBayes(labelCol="is_goal", featuresCol="features", smoothing=1.0)`). The `smoothing` parameter is used for Laplace smoothing to handle unseen features.

2. **Random Forest (RandomForestClassifier):**
   - **Usage in the Code:** It's used for classification (`RandomForestClassifier(labelCol="is_goal", featuresCol="features", numTrees=10)`). The `numTrees` parameter specifies the number of trees in the forest.

3. **Gradient Boosted Trees (GBTClassifier):**
   - **Usage in the Code:** It's used for classification (`GBTClassifier(labelCol="is_goal", featuresCol="features", maxDepth=5, maxIter=20)`). Parameters like `maxDepth` control the depth of individual trees, and `maxIter` specifies the number of boosting iterations.

4. **Support Vector Machine (LinearSVC):**
   - **Usage in the Code:** It's used for classification (`LinearSVC(labelCol="is_goal", featuresCol="features", maxIter=100)`). The `maxIter` parameter controls the maximum number of iterations.

The code then evaluates the performance of each algorithm using various metrics such as accuracy, precision, recall, F1-score, ROC AUC. These metrics help assess the quality of the classification models. The code uses the `BinaryClassificationEvaluator` from PySpark for ROC AUC evaluation.

***HYPER PARAMETER TUNING***
In the process of building our machine learning models, we recognized the importance of optimizing hyperparameters to enhance model performance. Hyperparameter tuning is a crucial step in the model development pipeline, as it allows us to systematically explore different configurations and select the set of hyperparameters that yields the best performance.
