# Machine Learning with Python
The [MLPy2021 repo](https://github.com/munich-ml/MLPy2021) contains the Jupyter notebooks, datasets and models for the **2021 "Machine Learning with Python" class** at [DHBW Friedrichshafen](https://www.ravensburg.dhbw.de/studienangebot/bachelor-studiengaenge/elektrotechnik-fahrzeugelektronik.html).

## Welcome to Python and Colab [MLPy2021_slides.pptx p2](https://github.com/munich-ml/MLPy2021/blob/master/MLPy2021_slides.pptx)
- running **Python** locally or in the cloud ([**Google Colab**](https://colab.research.google.com/) for this course) 
- Python [`scripts.py`](https://www.python.org/) vs. [`jupyter_notebooks.ipynb`](https://jupyter.org/)

# Epic 1: Python basics
Parsing logfiles is a regular task for most data scientists. **Epic 1** tackles such parsing-task, the **logfile challange**. It starts with Python's basic datatypes (`str`, `list`, `dict`,..) and contineously improves the parser by using higher level libraries (**NumPy**, **Matplotlib**, **Pandas**) and finally **classes**.  

## [10_Logfile_challenge.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/10_Logfile_challenge.ipynb)
- `!git clone https://github.com/munich-ml/MLPy2021/` for getting `logfile.csv`
- task: Parsing the logfile from `s`, `<type 'str'>`  
- `lines = s.split("\n")`
- `<type 'list'>` is iterable: `for line in lines:`
- `<type 'dict'>` for key-value lookup, like `idxs = {'header': 2, 'measurements': 8}`
- `<type 'set'>` for set operations like `union` or `diff`
- 2-dim `data` became a list of lists: `data[row][col]`, that doesn't support matrix operations like element-wise multiplication or arbitrary indexing.

## [11_NumPy.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/11_NumPy.ipynb)
**NumPy** (alias `np`) is a Python module made for matrix math.

The **`numpy.array`** is NumPy's standard datatype.
- `numpy.array`s may have 1, 2 or N dimensions
- all items have the same datatype
- provide arbitrary indexing
- provide element-wise operations

NumPy provides various *concatanation methods*
- `np.column_stack()` supports horizontal stacking of 2D and 1D arrays

## [12_Logfiles_w_NumPy.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/12_Logfiles_w_NumPy.ipynb)
Logfile challenge reworked to use NumPy arrays

## [13_Matplotlib.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/13_Matplotlib.ipynb)
For inspiration what and how to plot go to [Matplotlib website](https://matplotlib.org/)

The `%matplotlib` magic switches between:
- static inline plots (default) with `%matplotlib inline`
- interactive plots with `%matplotlib qt`

## [14_Pandas.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/14_Pandas.ipynb) 
**Pandas** (alias `pd`) is a Python module providing fast, flexible, and expressive data structures which are designed to make working with "labeled" data both easy and intuitive. 

**`pd.DataFrame`** is a two-dimensional data structure with labeled axes, rows (`index`) and `column`. The data is often a `np.array`.

**`pd.Series`** is the one-dimensional version of the `pd.DataFrame`.

## [15_Logfiles_w_Pandas.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/15_Logfiles_w_Pandas.ipynb)
Logfile challenge reworked to use Pandas DataFrames

## [17_object_oriented_programming.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/17_object_oriented_programming.ipynb)
Skipped in MLPy2021 class

## [18_Logfiles_w_classes.ipynb](https://github.com/munich-ml/MLPy2021/blob/0a5b6f12142b7f6822e46c3dfcfb4de30ab20fe1/18_Logfiles_w_classes.ipynb) 

The object oriented implementation of the *logfile challenge* brings some advantages:

**Functions (*methods*) and data bundled in one object**

**Cleaner interfaces and namespace**, e.g. 
> - `log = Log(fn)`
> - `log.plot(sig_names=["sig0_cal"])`            

versus

> - `log_data = parse_logfile_string_to_pandas(s)`
> - `plot(log_data, sig_names=["sig0_cal"])`

**overwrite operators** like `__str__`, `__lt__`, ...

# Epic 2: Machine Learning with Scikit-learn

## [21_first_machine_lerning_models.ipynb](https://github.com/munich-ml/MLPy2021/blob/92502102b3dddd20e48fd0550d22eaa1e9a63bb4/21_first_machine_lerning_models.ipynb)
- **Scikit-learn** offers standard interfaces to its models, e.g. `model.fit(x_train, y_train)`, `model.predict(x_new)`
- **RSME** or *root mean squared error* used as performance criterion for the **regression problem**
- a model is supposed to **generalize** the training data, not to **memorize** it
- an **overfitting** model performs much worse on the *test data* than on the *training RSME*
- the common root cause is **too few training data** applied to a **too complex model**
- **Regularization** helps to avoids overfitting of complex models
- an **underfitting** model performs bad on both data sets
- comparison of **Linear regression model** and **Decision tree model**

## [22_end2end_ml_project.ipynb](https://github.com/munich-ml/MLPy2021/blob/8942d1a0440c5b1c7d70eed5bf56f82839ab2d5f/22_end2end_ml_project.ipynb)
- `housing` dataset with 10 **attributes** and 20.640 samples
- `median_house_value` will be the *target attribute*, also called **label**. The other attributes will be the **features**
- the `median_house_value` distribution is odd, with an obvious cap at 500,000
- 9 attributes are **numerical**, 1 is **categorical**
- the `total_bedrooms` feature is incomplete, meaning it has *null* values
- Scikit-learn's **`train_test_split`** slits datasets randomly and reproducable into subsets (e.g. for training and test)
- **`StratifiedShuffleSplit`** ensures that the feature distributions are **representative** (w.r.t. a certain feature)
- Pandas **`corr`** and **`scatter_matrix`** are useful tools for **dataset exploration**
- **feature combinations** can be more informative that the native features. Implemented as custom **`CombinedAttrAdder`** class
- **missing values** need to be handled, e.g. `dropna`, `fillna` or better Scikit-learn's **`SimpleImputer`**
- **categorical text features** are encoded using the **`OneHotEncoder`**
- features are scaled using **`StandardScaler`**. Alternatives: `MinMaxScaler`, `QuantileTransformer`
- **transformation pipelines** execute multiple *transformers* as *one-liner*
- various models trained: Linear regressor, decision tree, random forest, SVM
- **validation dataset** used for model selection and hyperparameter tuning
- **cross validation** trains and validates a model k-times, each time with a differnet validation subset
- automized model tuning using **`GridSearchCV`** and **`RandomizedSearchCV`**. `CV` because *cross validation* is applied
- `RandomForestRegressor` provides **`feature_importances_`** to identify the most and least important features
- **final test** is the first and only time, the *testset* should be used!
- concluding [SciKit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

# Epic 3: Neural Networks for Computer Vision with Tensorflow
## Neural networks [MLPy2021_slides.pptx p6..8](https://github.com/munich-ml/MLPy2021/blob/master/MLPy2021_slides.pptx)
- **layers** (input, hidden and output), weights and biases
- neural net training means **finding weights & biases that minimize the cost function**
- (error) **back propagation** modifies the weights accrding to their effect on the error
- **epoch** is the training over the full dataset, executed in batches
- get an intuition for NN's with [TensorFlow playground](https://playground.tensorflow.org)

## [31_fMNIST_classifier_keras.ipynb](https://github.com/munich-ml/MLPy2021/blob/8942d1a0440c5b1c7d70eed5bf56f82839ab2d5f/31_fMNIST_classifier_keras.ipynb)
- `fashion_MNIST` classification problem: 60,000 images 28x28 pixels from 10 classes
- images as NumPy arrays. Plotting with `plt.imshow(X_train[img])`
- **build the model** with `keras.models.Sequential()`
- **compile the model** and handover `loss`, `optimizer` and `metrics`
- **train the model** and don't forget to chose **GPU** as hardware accelerator in Runtime 
- mount Google Drive and save the model (`model.save()`) and the data (`picke_out()`)

## [32_evaluate_fMNIST_classifier.ipynb](https://github.com/munich-ml/MLPy2021/blob/8942d1a0440c5b1c7d70eed5bf56f82839ab2d5f/32_evaluate_fMNIST_classifier.ipynb)
- load the model from Google Drive `keras.models.load_model()`
- predict instances from the *validation data set*
- **confusion matrix** `tf.math.confusion_matrix`, and how to plot it
- performance measures for *classifiers*, *accuracy*, *precision, recall*, *F1-score*
- discussion of prediction examples, in particular `false_pos`'s and `false_neg`'s

## Convolutional neural networks [MLPy2021_slides.pptx p9..11](https://github.com/munich-ml/MLPy2021/blob/master/MLPy2021_slides.pptx)
- disadvantages of **dense neural networks** for **computer vision** (scaleability, prone to overfitting, semantic neighborhood, position dependence)
- CNN layer types (convolutional layers, pooling layers, dense layers)
- lower layers learn basic shapes, while higher layers learn more complete objects (**hint to transfer learning**)
- get some intuition for CNN's with [Adam Harleys's "Interactive CNN Visualizer"](https://www.cs.ryerson.ca/~aharley/vis/conv/)

## [34_fMNIST_with_CNNs.ipynb](https://github.com/munich-ml/MLPy2021/blob/bcbe43d639a6214aff6aad5c096f77b9be79b7f3/34_fMNIST_with_CNNs.ipynb)
This notebook is very similar to `31_fMNIST_classifier_keras.ipynp`, except for:
- slightly different preprocessing (scaling to std. diviation and 4dim image arrays)
- build the model with CNN layers (e.g. `keras.layers.Conv2D` and `keras.layers.MaxPooling2D`)

## [35_batch_evaluate_fMNIST.ipynb](https://github.com/munich-ml/MLPy2021/blob/488f41f731e71f33a722c296e096768a1851b912/35_batch_evaluate_fMNIST.ipynb)
- mount Google Drive and search all models in `models` subdir
- benchmark all models w.r.t. **accuracy** and **execution time**
- **concluding CNNs** (e.g. for MNIST the CNNs don't benefit from their position independency)

# Appendix

## Progmming language popularity
- [TOIBE popularity index](https://www.tiobe.com/tiobe-index/) ratings based on search quantities of 25 engines (Google, Baidu,.. but also Wikipedia)
> 1. C
> 2. **Python** 
> 3. Java

- [PYPL](http://pypl.github.io/) measures how often language tutorials are googled by exploring Google Trends.
> 1. **Python** 
> 2. Java
> 3. JavaScript

- [GitHub statistics](https://madnight.github.io/githut/#/pull_requests/2020/1) percentage pull-requests / commits / issues on GitHub.
> 1. JavaScript
> 2. **Python** 

- [Stackoverflow survey 2019](https://insights.stackoverflow.com/survey/2019): Key result #1: *Python, the fastest-growing major programming language, has risen in the ranks of programming languages in our survey yet again, edging out Java this year and standing as the second most loved language (behind Rust).*
> 1. JavaScript
> 2. HTML/CSS
> 3. SQL
> 4. **Python** 

## Further reading and exercise
- Harrison Kinsley's [PythonProgramming.net](https://pythonprogramming.net/) is a tutorials site with video & text based tutorials for Python programming.
- Corey Schafer's [YouTube channel](https://www.youtube.com/user/schafer5)
- [Kaggle](https://www.kaggle.com/), the data science community with datasets, notebooks, courses and competition
- Michael Kennedy's [talk Python to me](https://talkpython.fm/) podcast
-  Aurelien Geron _Hands-on Machine Learning with Scikit-learn, Keras & TensorFlow_ [Book on Amazon](https://www.amazon.de/Aur%C3%A9lien-G%C3%A9ron/dp/1492032646/ref=sr_1_3?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&dchild=1&keywords=Hands-on+Machine+Learning+with+Scikit-learn%2C+Keras+%26+TensorFlow%2C+Aurelien+Geron%2C&qid=1589875241&sr=8-3)
- Andreas Mueller: _Introduction to Machine Learning with Python_ [Book on Amazon](https://www.amazon.de/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- Andreas Mueller: [Applied ML spring semester 2020](https://www.cs.columbia.edu/~amueller/comsw4995s20/), with videos, slides and notebooks
