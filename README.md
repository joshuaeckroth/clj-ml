# clj-ml

A machine learning library for Clojure built on top of Weka and friends.

## Installation

### Installing from Clojars

    [cc.artifice/clj-ml "0.6.0-SNAPSHOT"]

## Supported algorithms

 * Filters
   * Discretization (supervised, unsupervised, PKI)
   * Nominal to binary (supervised, unsupervised)
   * Numeric to nominal
   * String to word vector
   * Attribute manipulation (reorder, add, remove range, remove percentage, etc.)
   * Resample (supervised, unsupervised)
   * Replace missing values with mean (numeric attributes) or mode (nominal attributes)

 * Classifiers
   * k-Nearest neighbor
   * Decision trees: C4.5/J4.8, Boosted stump, Random forest, Rotation forest, M5P
   * Naive Bayes
   * Multilayer perceptrons
   * Support vector machines (grid-based training), SMO, Spegasos

 * Regression
   * Linear
   * Logistic
   * Pace
   * Additive gradient boosting

 * Clusterers
   * k-Means
   * Cobweb
   * Expectation-maximization

## Usage

API documenation can be found [here](http://clj-ml.artifice.cc/doc/index.html).

### I/O of data

```clojure
user> (use 'clj-ml.io)
nil

user> (def ds (load-instances :arff "file:///home/josh/git/clj-ml/iris.arff"))
#'user/ds
user> ds
#<Instances @relation iris

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric
@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}

@data
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
...

user> (def ds (load-instances :arff "http://repository.seasr.org/Datasets/UCI/arff/iris.arff"))
#'user/ds

user> (save-instances :csv "iris.csv" ds)
nil
user> (println (slurp "iris.csv"))
sepallength,sepalwidth,petallength,petalwidth,class
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
...

user> (def ds (load-instances :csv "file:///home/josh/git/clj-ml/iris.csv"))
#'user/ds
user> ds
#<Instances @relation stream

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric
@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}

@data
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5,3.4,1.5,0.2,Iris-setosa
```

### Working with datasets

```clojure
user> (use 'clj-ml.data)
nil

user> (def ds (make-dataset"my-name" [:length :width {:style nil} {:kind [:good :bad]}]
                            [[12 24 "longish" :good]
                             [8 5 "shortish" :bad]]))
#'user/ds
user> ds
#<ClojureInstances @relation my-name

@attribute length numeric
@attribute width numeric
@attribute style string
@attribute kind {good,bad}

@data
12,24,longish,good
8,5,shortish,bad>

user> (dataset-seq ds)
(#<Instance 12,24,longish,good> #<Instance 8,5,shortish,bad>)

user> (map instance-to-map (dataset-seq ds))
({:kind :good, :style "longish", :width 24.0, :length 12.0}
{:kind :bad, :style "shortish", :width 5.0, :length 8.0})

user> (map instance-to-vector (dataset-seq ds))
([12.0 24.0 "longish" :good] [8.0 5.0 "shortish" :bad])
```

### Filtering datasets

```clojure
user> (use 'clj-ml.filters 'clj-ml.io)
nil

user> (def ds (load-instances :csv "file:///home/josh/git/clj-ml/iris.csv"))
#'user/ds

user> (def discretize (make-filter :unsupervised-discretize
                                   {:dataset-format ds
                                    :attributes [:sepallength :petallength]}))
#'user/discretize

user> (def filtered-ds (filter-apply discretize ds))
#'user/filtered-ds

user> (map instance-to-map (dataset-seq filtered-ds))
({:class :Iris-setosa, :petalwidth 0.2, :petallength :'(-inf-1.59]',
 :sepalwidth 3.5, :sepallength :'(5.02-5.38]'}
{:class :Iris-setosa, :petalwidth 0.2, :petallength :'(-inf-1.59]',
 :sepalwidth 3.0, :sepallength :'(4.66-5.02]'}
{:class :Iris-setosa, :petalwidth 0.2, :petallength :'(-inf-1.59]',
 :sepalwidth 3.2, :sepallength :'(4.66-5.02]'}
{:class :Iris-setosa, :petalwidth 0.2, :petallength :'(-inf-1.59]',
 :sepalwidth 3.1, :sepallength :'(-inf-4.66]'}
{:class :Iris-setosa, :petalwidth 0.2, :petallength :'(-inf-1.59]',
 :sepalwidth 3.6, :sepallength :'(4.66-5.02]'}
...) ;; the petallength and sepallength attributes are now nominal
```

Equivalently,

```clojure
user> (def filtered-ds (->> "file:///home/josh/git/clj-ml/iris.csv"
                            (load-instances :csv)
                            (make-apply-filter :unsupervised-discretize
                                               {:attributes [:sepallength :petallength]})))
```

### Using classifiers

```clojure
user> (use 'clj-ml.classifiers 'clj-ml.data 'clj-ml.utils)
nil

user> (def ds (-> (load-instances :arff "file:///home/josh/git/clj-ml/iris.arff")
                  (dataset-set-class :class)))
#'user/ds

user> (def classifier (-> (make-classifier :decision-tree :c45)
                          (classifier-train ds)))
#'user/classifier

user> (def instance (-> (first (dataset-seq ds))
                        (instance-set-class-missing)))

user> (classifier-classify classifier instance)
:Iris-setosa
```

Evaluation:

```clojure
user> (def evaluation (classifier-evaluate classifier :cross-validation ds 10))
#'user/evaluation

user> (clojure.pprint/pprint (dissoc evaluation :summary :confusion-matrix))
{:incorrect 7.0,
 :root-relative-squared-error 36.693518966642074,
 :sf-entropy-gain -4076.3670930399717,
 :recall
 {:Iris-setosa 0.9795918367346939,
  :Iris-versicolor 0.94,
  :Iris-virginica 0.94},
 :kb-information 217.7935138195151,
 :kb-relative-information 13741.240800360849,
 :false-positive-rate
 {:Iris-setosa 0.0,
  :Iris-versicolor 0.04040404040404041,
  :Iris-virginica 0.030303030303030304},
 :percentage-correct 95.30201342281879,
 :roc-area
 {:Iris-setosa 0.984845423317842,
  :Iris-versicolor 0.9456,
  :Iris-virginica 0.9496},
 :kb-mean-information 1.4617014350303028,
 :percentage-unclassified 0.0,
 :percentage-incorrect 4.697986577181208,
 :root-mean-squared-error 0.17297908222448935,
 :unclassified 0.0,
 :correlation-coefficient
 {:nan "Can't compute correlation coefficient: class is nominal!"},
 :correct 142.0,
 :sf-mean-entropy-gain -27.358168409664238,
 :mean-absolute-error 0.04083212821368881,
 :relative-absolute-error 9.187228848079984,
 :error-rate 0.04697986577181208,
 :kappa 0.9295222650179066,
 :f-measure
 {:Iris-setosa 0.9896907216494846,
  :Iris-versicolor 0.9306930693069307,
  :Iris-virginica 0.94},
 :false-negative-rate
 {:Iris-setosa 0.02040816326530612,
  :Iris-versicolor 0.06,
  :Iris-virginica 0.06},
 :evaluation-object #<Evaluation weka.classifiers.Evaluation@6a7272ca>,
 :average-cost 0.0,
 :precision
 {:Iris-setosa 1.0,
  :Iris-versicolor 0.9215686274509803,
  :Iris-virginica 0.94}}

user> (println (:summary evaluation))

Correctly Classified Instances         142               95.302  %
Incorrectly Classified Instances         7                4.698  %
Kappa statistic                          0.9295
Mean absolute error                      0.0408
Root mean squared error                  0.173 
Relative absolute error                  9.1872 %
Root relative squared error             36.6935 %
Total Number of Instances              149     
Ignored Class Unknown Instances                  1     

nil
user> (println (:confusion-matrix evaluation))
=== Confusion Matrix ===

  a  b  c   <-- classified as
 48  1  0 |  a = Iris-setosa
  0 47  3 |  b = Iris-versicolor
  0  3 47 |  c = Iris-virginica

nil
```
Saving and restoring (trained) classifiers:

```clojure

user> (serialize-to-file classifier "my-classifier.bin")
"my-classifier.bin"

user> (def classifier2 (deserialize-from-file "my-classifier.bin"))
#'user/classifier2

user> (classifier-classify classifier2 instance)
:Iris-setosa
```

Text document handling:

```clojure
user> (def docs [{:title "Document title 1"
                  :fulltext "This is the fulltext..."
                  :terms {"Topic" ["Sports"]}}
                 {:title "Another document title"
                  :fulltext "Some more \"fulltext\"; rabbit artificial machine bananas"
                  :terms {"Topic" ["Politics" "Food"]}}])
#'user/docs

user> (docs-to-dataset docs "Topic" "Sports" 1 "/tmp" :stemmer true :lowercase false)
#<Instances @relation 'docs-weka.filters.unsupervised.attribute.StringToWordVector...'

@attribute class {no,yes}
@attribute title-1 numeric
@attribute title-Another numeric
@attribute title-Document numeric
@attribute title-document numeric
@attribute title-titl numeric
@attribute fulltext-Some numeric
@attribute fulltext-This numeric
@attribute fulltext-artifici numeric
@attribute fulltext-banana numeric
@attribute fulltext-fulltext numeric
@attribute fulltext-is numeric
@attribute fulltext-machin numeric
@attribute fulltext-more numeric
@attribute fulltext-rabbit numeric
@attribute fulltext-the numeric

@data
{0 yes,1 0.480453,3 0.480453,7 0.480453,11 0.480453,15 0.480453}
{2 0.480453,4 0.480453,6 0.480453,8 0.480453,9 0.480453,12 0.480453,13 0.480453,14 0.480453}>
user>
```

Words appearing in the dataset will only be those appearing in the
documents (or a subset; by default, the most common 1000 words). This
presents a problem when new documents are loaded and used in a
classifier trained on other documents. The classifier will not know
how to handle word attributes that were not present in the training
set.

The `docs-to-dataset` function provides the ability to save the
training documents dataset and "filter" the testing documents through
this dataset to ensure the same word attributes are extracted for both
sets. The following example shows that the words "foo, bar, baz, quux"
are ignored in the new (testing) documents, and all the original
attributes in the training dataset are retained.

```clojure
user> (docs-to-dataset docs "Topic" "Sports" 1 "/tmp"
                       :stemmer true :lowercase false :training true)
#<Instances @relation 'docs-weka.filters.unsupervised.attribute.StringToWordVector...'

@attribute class {no,yes}
@attribute title-1 numeric
@attribute title-Another numeric
@attribute title-Document numeric
@attribute title-document numeric
@attribute title-titl numeric
@attribute fulltext-Some numeric
@attribute fulltext-This numeric
@attribute fulltext-artifici numeric
@attribute fulltext-banana numeric
@attribute fulltext-fulltext numeric
@attribute fulltext-is numeric
@attribute fulltext-machin numeric
@attribute fulltext-more numeric
@attribute fulltext-rabbit numeric
@attribute fulltext-the numeric

@data
{2 0.480453,4 0.480453,6 0.480453,8 0.480453,9 0.480453,12 0.480453,13 0.480453,14 0.480453}
{0 yes,1 0.480453,3 0.480453,7 0.480453,11 0.480453,15 0.480453}>

user> (def docs2 [{:title "Document title 1 foo bar"
                   :fulltext "baz rabbit quux"
                   :terms {"Topic" ["Sports"]}}])
#'user/docs2

user> (docs-to-dataset docs2 "Topic" "Sports" 1 "/tmp"
                       :stemmer true :lowercase false :testing true)
#<Instances @relation 'docs-weka.filters.unsupervised.attribute.StringToWordVector...'

@attribute class {no,yes}
@attribute title-1 numeric
@attribute title-Another numeric
@attribute title-Document numeric
@attribute title-document numeric
@attribute title-titl numeric
@attribute fulltext-Some numeric
@attribute fulltext-This numeric
@attribute fulltext-artifici numeric
@attribute fulltext-banana numeric
@attribute fulltext-fulltext numeric
@attribute fulltext-is numeric
@attribute fulltext-machin numeric
@attribute fulltext-more numeric
@attribute fulltext-rabbit numeric
@attribute fulltext-the numeric

@data
{0 yes,1 0.480453,3 0.480453,14 0.480453}>
user> 
```

### Using clusterers

```clojure
user> (use 'clj-ml.clusterers)
nil

user> (def ds (-> (load-instances :arff "file:///home/josh/git/clj-ml/iris.arff")
                  (dataset-remove-attribute-at 4)))
#'user/ds
user> ds
#<Instances @relation iris

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric

@data
5.1,3.5,1.4,0.2
4.9,3,1.4,0.2
4.7,3.2,1.3,0.2
4.6,3.1,1.5,0.2
5,3.6,1.4,0.2
5.4,3.9,1.7,0.4
4.6,3.4,1.4,0.3
...

user> (def clusterer (make-clusterer :k-means {:number-clusters 3}))
#'user/clusterer

user> (clusterer-build clusterer ds)
nil

user> clusterer
#<SimpleKMeans 
kMeans
======

Number of iterations: 6
Within cluster sum of squared errors: 6.998114004826762
Missing values globally replaced with mean/mode

Cluster centroids:
                           Cluster#
Attribute      Full Data          0          1          2
                   (150)       (61)       (50)       (39)
=========================================================
sepallength       5.8433     5.8885      5.006     6.8462
sepalwidth         3.054     2.7377      3.418     3.0821
petallength       3.7587     4.3967      1.464     5.7026
petalwidth        1.1987      1.418      0.244     2.0795


>

user> (def clustered-ds (clusterer-cluster clusterer ds))
#'user/clustered-ds

user> clustered-ds
#<ClojureInstances @relation 'clustered iris'

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric
@attribute class {0,1,2}

@data
5.1,3.5,1.4,0.2,1
4.9,3,1.4,0.2,1
4.7,3.2,1.3,0.2,1
4.6,3.1,1.5,0.2,1
5,3.6,1.4,0.2,1
5.4,3.9,1.7,0.4,1
4.6,3.4,1.4,0.3,1
5,3.4,1.5,0.2,1
4.4,2.9,1.4,0.2,1
...
```

## Example: Home price prediction

http://www.ibm.com/developerworks/library/os-weka1/

```clojure
user> (def homes (make-dataset "homes" [:house-size :lot-size :bedrooms
                                        :granite :bathroom :sellingPrice]
                               [[3529, 9191, 6, 0, 0, 205000]
                                [3247, 10061, 5, 1, 1, 224900]
                                [4032, 10150, 5, 0, 1, 197900]
                                [2397, 14156, 4, 1, 0,189900]
                                [2200, 9600, 4, 0, 1, 195000] 
                                [3536, 19994, 6, 1, 1,325000] 
                                [2983, 9365, 5, 0, 1, 230000]]))
#'user/homes

user> (def homes (dataset-set-class homes :sellingPrice))
#'user/homes

user> homes
#<ClojureInstances @relation homes

@attribute house-size numeric
@attribute lot-size numeric
@attribute bedrooms numeric
@attribute granite numeric
@attribute bathroom numeric
@attribute sellingPrice numeric

@data
3529,9191,6,0,0,205000
3247,10061,5,1,1,224900
4032,10150,5,0,1,197900
2397,14156,4,1,0,189900
2200,9600,4,0,1,195000
3536,19994,6,1,1,325000
2983,9365,5,0,1,230000>

user> (def reg (classifier-train (make-classifier :regression :linear) homes))
#'user/reg

user> reg
#<LinearRegression 
Linear Regression Model

sellingPrice =

    -26.6882 * house-size +
      7.0551 * lot-size +
  43166.0767 * bedrooms +
  42292.0901 * bathroom +
 -21661.1208>
user> 

user> (classifier-predict-numeric reg (make-instance homes [3198, 9669, 5, 1, 1, nil]))
219328.35717359098
```

## Example: Predicting survival on the Titanic

https://www.kaggle.com/c/titanic-gettingStarted

First globally replace all double quoted strings `""foo""` with
backslash quoted strings: `\"foo\"`. Weka does not handle the former.

```clojure
user> (def titanicds (load-instances :csv "file:///home/josh/git/clj-ml/titanic-train.csv"))
user> titanicds
#<Instances @relation stream

@attribute PassengerId numeric
@attribute Survived numeric
@attribute Pclass numeric
@attribute Name {'Braund, Mr. Owen Harris','Cumings, Mrs. John Bradley (Florence Briggs Thayer)', ...}
@attribute Sex {male,female}
@attribute Age numeric
@attribute SibSp numeric
@attribute Parch numeric
@attribute Ticket {'A/5 21171','PC 17599','STON/O2. 3101282',113803.0, ...}
@attribute Fare numeric
@attribute Cabin {C85,C123,E46,G6,C103,D56,A6,'C23 C25 C27', ...}
@attribute Embarked {S,C,Q}

@data
1,0,3,'Braund, Mr. Owen Harris',male,22,1,0,'A/5 21171',7.25,?,S
2,1,1,'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',female,38,1,0,'PC 17599',71.2833,C85,C
3,1,3,'Heikkinen, Miss. Laina',female,26,0,0,'STON/O2. 3101282',7.925,?,S
4,1,1,'Futrelle, Mrs. Jacques Heath (Lily May Peel)',female,35,1,0,113803.0,53.1,C123,S
5,0,3,'Allen, Mr. William Henry',male,35,0,0,373450.0,8.05,?,S
6,0,3,'Moran, Mr. James',male,?,0,0,330877.0,8.4583,?,Q
7,0,1,'McCarthy, Mr. Timothy J',male,54,0,0,17463.0,51.8625,E46,S
8,0,3,'Palsson, Master. Gosta Leonard',male,2,3,1,349909.0,21.075,?,S
9,1,3,'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',female,27,0,2,347742.0,11.1333,?,S
10,1,2,'Nasser, Mrs. Nicholas (Adele Achem)',female,14,1,0,237736.0,30.0708,?,C
11,1,3,'Sandstrom, Miss. Marguerite Rut',female,4,1,1,'PP 9549',16.7,G6,S
...
>

#'user/titanicds
user> (def titanicds (dataset-set-class titanicds :Survived))
#'user/titanicds
user> (dataset-class-index titanicds)
1

user> (def titanicds (make-apply-filter :numeric-to-nominal
                                        {:attributes [:Survived]}
                                        titanicds))
#'user/titanicds
user> titanicds
#<Instances @relation stream-weka.filters.unsupervised.attribute.NumericToNominal-R2

@attribute PassengerId numeric
@attribute Survived {0,1}
@attribute Pclass numeric
...
>

user> (def titanicds (make-apply-filter :replace-missing-values {} titanicds))

user> (def titanicds (make-apply-filter :remove-attributes
                                        {:attributes [:PassengerId :Name :Ticket :Cabin]}
                                        titanicds))
#'user/titanicds
user> titanicds
#<Instances @relation 'stream-weka.filters.unsupervised.attribute.NumericToNominal...'

@attribute Survived {0,1}
@attribute Pclass numeric
@attribute Sex {male,female}
@attribute Age numeric
@attribute SibSp numeric
@attribute Parch numeric
@attribute Fare numeric
@attribute Embarked {S,C,Q}

@data
0,3,male,22,1,0,7.25,S
1,1,female,38,1,0,71.2833,C
1,3,female,26,0,0,7.925,S
1,1,female,35,1,0,53.1,S
0,3,male,35,0,0,8.05,S
0,3,male,?,0,0,8.4583,Q
...
>

user> (dataset-class-index titanicds)
0
user> (dataset-class-name titanicds)
:Survived

user> (def evaluation (classifier-evaluate (make-classifier :decision-tree :random-forest)
                                           :cross-validation titanicds 10))
#'user/evaluation
user> (println (:summary evaluation))

Correctly Classified Instances         727               81.5937 %
Incorrectly Classified Instances       164               18.4063 %
Kappa statistic                          0.6039
Mean absolute error                      0.2409
Root mean squared error                  0.3819
Relative absolute error                 50.9302 %
Root relative squared error             78.532  %
Total Number of Instances              891     

nil
user> (println (:confusion-matrix evaluation))
=== Confusion Matrix ===

   a   b   <-- classified as
 483  66 |   a = 0
  98 244 |   b = 1

nil
```

Ok, looks good, let's try training on the full training data and
testing on the testing data.

```clojure
user> (def titanic-testds (load-instances :csv "file:///home/josh/git/clj-ml/titanic-test.csv"))

user> (def titanic-test-passids (map (comp int :PassengerId)
                                     (dataset-as-maps titanic-testds)))
#'user/titanic-test-passids
user> titanic-test-passids
(892 893 894 895 896 897 898 899 900 ...)

user> (def titanic-testds (->> titanic-testds
                               (make-apply-filter :remove-attributes
                                                  {:attributes [:PassengerId :Name :Ticket :Cabin]})
                               (make-apply-filter :replace-missing-values {})
                               (make-apply-filter :add-attribute
                                                  {:type :nominal :name :Survived
                                                   :column 0 :labels ["0" "1"]})))
#'user/titanic-testds

user> (def titanic-testds (dataset-set-class titanic-testds :Survived))
#'user/titanic-testds

user> titanic-testds
#<Instances @relation 'stream-weka.filters.unsupervised.attribute.Remove...'

@attribute Survived {0,1}
@attribute Pclass numeric
@attribute Sex {male,female}
@attribute Age numeric
@attribute SibSp numeric
@attribute Parch numeric
@attribute Fare numeric
@attribute Embarked {Q,S,C}

@data
?,3,male,34.5,0,0,7.8292,Q
?,3,female,47,1,0,7,S
?,2,male,62,0,0,9.6875,Q
?,3,matitanle,27,0,0,8.6625,S
?,3,female,22,1,1,12.2875,S
?,3,male,14,0,0,9.225,S
?,3,female,30,0,0,7.6292,Q
...
>

user> (def classifier (classifier-train (make-classifier :decision-tree :random-forest) titanicds))
#'user/classifier

user> (def preds (for [instance (dataset-seq titanic-testds)]
                      (name (classifier-classify classifier instance))))
#'user/preds

user> preds
("0" "1" "0" "0" "0" "0" "1" "0" "0" "0" ...)

#'user/preds

user> (spit "titanic-predictions.csv"
            (clojure.string/join "\n" (cons "Survived,PassengerId"
                                            (map (fn [c1 c2] (format "%s,%d" c1 c2))
                                                 preds titanic-test-passids))))
nil

user> (println (slurp "titanic-predictions.csv"))
Survived,PassengerId
0,892
1,893
0,894
0,895
0,896
0,897
1,898
0,899
0,900
0,901
0,902
...
```

## Thanks YourKit!

YourKit is kindly supporting open source projects with its
full-featured Java Profiler.  YourKit, LLC is the creator of
innovative and intelligent tools for profiling Java and .NET
applications. Take a look at YourKit's leading software products: <a
href="http://www.yourkit.com/java/profiler/index.jsp">YourKit Java
Profiler</a> and <a
href="http://www.yourkit.com/.net/profiler/index.jsp">YourKit .NET
Profiler</a>.

## License

MIT License

## Authors

* 2010: [Antonio Garrote](https://github.com/antoniogarrote)
* 2010-2013: [Ben Mabey](https://github.com/bmabey)
* 2013: [Joshua Eckroth](https://github.com/joshuaeckroth)
