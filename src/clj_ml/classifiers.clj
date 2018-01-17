;;
;; Classifiers
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
      clj-ml.classifiers
  "This namespace contains several functions for building classifiers using different
   classification algorithms: Bayes networks, multilayer perceptron, decision tree or
   support vector machines are available. Some of these classifiers have incremental
   versions so they can be built without having all the dataset instances in memory.

   Functions for evaluating the classifiers built using cross validation or a training
   set are also provided.

   A sample use of the API for classifiers is shown below:

    (use 'clj-ml.classifiers)

    ; Building a classifier using a  C4.5 decision tree
    (def *classifier* (make-classifier :decision-tree :c45))

    ; We set the class attribute for the loaded dataset.
    ; *dataset* is supposed to contain a set of instances.
    (dataset-set-class *dataset* 4)

    ; Training the classifier
    (classifier-train *classifier* *dataset*)

    ; We evaluate the classifier using a test dataset
    (def *evaluation*   (classifier-evaluate *classifier* :dataset *dataset* *trainingset*))

    ; We retrieve some data from the evaluation result
    (:kappa *evaluation*)
    (:root-mean-squared-error *evaluation*)
    (:precision *evaluation*)

    ; A trained classifier can be used to classify new instances
    (def *to-classify* (make-instance *dataset*  {:class :Iris-versicolor
                                                  :petalwidth 0.2
                                                  :petallength 1.4
                                                  :sepalwidth 3.5
                                                  :sepallength 5.1}))

    ; We retrieve the index of the class value assigned by the classifier
    (classifier-classify *classifier* *to-classify*)

    ; We retrieve a symbol with the value assigned by the classifier
    ; and assigns it to a certain instance
    (classifier-label *classifier* *to-classify*)

   A classifier can also be trained using cross-validation:

    (classifier-evaluate *classifier* :cross-validation *dataset* 10)

   Finally a classifier can be stored in a file for later use:

    (use 'clj-ml.utils)

    (serialize-to-file *classifier*
     \"/Users/antonio.garrote/Desktop/classifier.bin\")
"
  (:use [clj-ml utils data kernel-functions options-utils])
  (:import (java.util Date Random)
           (weka.core Instance Instances)
           (weka.classifiers.lazy IBk)
           (weka.classifiers.trees J48 RandomForest M5P)
           (weka.classifiers.meta LogitBoost AdditiveRegression RotationForest RacedIncrementalLogitBoost 
                                  Bagging RandomSubSpace Stacking CostSensitiveClassifier MetaCost)
           (weka.classifiers.bayes NaiveBayes NaiveBayesUpdateable)
           (weka.classifiers.functions MultilayerPerceptron SMO LinearRegression Logistic PaceRegression SPegasos LibSVM PLSClassifier)
           (weka.classifiers AbstractClassifier Classifier Evaluation)))

;; Setting up classifier options

(defmulti #^{:skip-wiki true}
            make-classifier-options
  "Creates the right parameters for a classifier. Returns the parameters as a Clojure vector."
  (fn [kind algorithm map] [kind algorithm]))

(defmethod make-classifier-options [:lazy :ibk]
  ([kind algorithm m]
   (->> (check-options m
                       {:inverse-weighted "-I"
                        :similarity-weighted "-F"
                        :no-normalization "-N"})
        (check-option-values m
                             {:num-neighbors "-K"}))))

(defmethod make-classifier-options [:decision-tree :c45]
  ([kind algorithm m]
   (->> (check-options m
                       {:unpruned "-U"
                        :reduced-error-pruning "-R"
                        :only-binary-splits "-B"
                        :no-raising "-S"
                        :no-cleanup "-L"
                        :laplace-smoothing "-A"})
        (check-option-values m
                             {:pruning-confidence "-C"
                              :minimum-instances "-M"
                              :pruning-number-folds "-N"
                              :random-seed "-Q"}))))

(defmethod make-classifier-options [:bayes :naive]
  ([kind algorithm m]
   (check-options m
                  {:kernel-estimator "-K"
                   :supervised-discretization "-D"
                   :old-format "-O"})))

(defmethod make-classifier-options [:neural-network :multilayer-perceptron]
  ([kind algorithm m]
   (->> (check-options m
                       {:no-nominal-to-binary "-B"
                        :no-numeric-normalization "-C"
                        :no-normalization "-I"
                        :no-reset "-R"
                        :learning-rate-decay "-D"})
        (check-option-values m
                             {:learning-rate "-L"
                              :momentum "-M"
                              :epochs "-N"
                              :percentage-validation-set "-V"
                              :random-seed "-S"
                              :threshold-number-errors "-E"
                              :hidden-layers-string "-H"}))))

(defmethod make-classifier-options [:support-vector-machine :smo]
  ([kind algorithm m]
   (->> (check-options m {:fit-logistic-models "-M"})
        (check-option-values m
                             {:complexity-constant "-C"
                              :normalize "-N"
                              :tolerance "-L"
                              :epsilon-roundoff "-P"
                              :folds-for-cross-validation "-V"
                              :random-seed "-W"}))))

(defmethod make-classifier-options [:support-vector-machine :spegasos]
  ([kind algorithm m]
   (->> (check-options m {:no-normalization "-N"
                          :no-replace-missing "-M"})
        (check-option-values m
                             {:loss-fn "-F"
                              :epochs "-E"
                              :lambda "-L"}))))

(defmethod make-classifier-options [:support-vector-machine :libsvm]
  ([kind algorithm m]
   (->> (check-options m {:normalization "-Z"
                          :no-nominal-to-binary "-J"
                          :no-missing-value-replacement "-V"
                          :no-shrinking-heuristics "-H"
                          :probability-estimates "-B"})
        (check-option-values m
                             {:svm-type "-S"
                              :kernel-type "-K"
                              :kernel-degree "-D"
                              :kernel-gamma "-G"
                              :kernel-coef0 "-R"
                              :param-C "-C"
                              :param-nu "-N"
                              :loss-epsilon "-P"
                              :memory-cache "-M"
                              :tolerance-of-termination "-E"
                              :class-weight "-W"
                              :random-seed "-seed"}))))

(defmethod make-classifier-options [:support-vector-machine :libsvm-grid]
  ([kind algorithm m]
   (->> (check-options m {:normalization "-Z"
                          :no-nominal-to-binary "-J"
                          :no-missing-value-replacement "-V"
                          :no-shrinking-heuristics "-H"
                          :probability-estimates "-B"})
        (check-option-values m
                             {:svm-type "-S"
                              :kernel-type "-K"
                              :kernel-degree "-D"
                              :kernel-coef0 "-R"
                              :param-nu "-N"
                              :loss-epsilon "-P"
                              :memory-cache "-M"
                              :tolerance-of-termination "-E"
                              :class-weight "-W"
                              :random-seed "-seed"}))))

(defmethod make-classifier-options [:regression :linear]
  ([kind algorithm m]
   (->> (check-options m {:debug "-D"
                          :keep-colinear "-C"})
        (check-option-values m
                             {:attribute-selection "-S"
                              :ridge "-R"}))))

(defmethod make-classifier-options [:regression :logistic]
  ([kind algorithm m]
   (->> (check-options m {:debug "-D"})
        (check-option-values m
                             {:max-iterations "-S"
                              :ridge "-R"}))))

(defmethod make-classifier-options [:regression :pace]
  ([kind algorithm m]
   (->> (check-options m {:debug "-D"})
        (check-option-values m
                             {:shrinkage "-S"
                              :estimator "-E"}))))

(defmethod make-classifier-options [:regression :boosted-regression]
  ([kind algorithm m]
   (->> (check-options m {:debug "-D"})
        (check-option-values m
                             {:threshold "-S"
                              :num-iterations "-I"
                              :weak-learning-class "-W"}))))

(defmethod make-classifier-options [:decision-tree :boosted-stump]
  ([kind algorithm m]
   (->> (check-options m {:debug "-D"
                          :resampling "-Q"})
        (check-option-values m
                             {:weak-learning-class "-W"
                              :num-iterations "-I"
                              :random-seed "-S"
                              :percentage-weight-mass "-P"
                              :log-likelihood-improvement-threshold "-L"
                              :z-max-threshold-for-responses "-Z"
                              :shrinkage-parameter "-H"}))))

(defmethod make-classifier-options [:decision-tree :random-forest]
  ([kind algorithm m]
   (->>
    (check-options m {:debug "-D"
                      :break-ties "-B"
                      :print-classifiers "-print"
                      })
    (check-option-values m
                         {:num-trees-in-forest "-I"
                          :num-features-to-consider "-K"
                          :random-seed "-S"
                          :depth "-depth"
                          :size-of-bag "-P"
                          :parallelism "-num-slots" 
                          :min-num-instance-per-leaf "-M"
                          :min-variance-for-split "-V"
                          }))))

(defmethod make-classifier-options [:decision-tree :rotation-forest]
  ([kind algorithm m]
   (->>
    (check-options m {:debug "-D"})
    (check-option-values m
                         {:num-iterations "-I"
                          :use-number-of-groups "-N"
                          :min-attribute-group-size "-G"
                          :max-attribute-group-size "-H"
                          :percentage-of-instances-to-remove "-P"
                          :filter "-F"
                          :random-seed "-S"
                          :weak-learning-class "-W"}))))

(defmethod make-classifier-options [:decision-tree :m5p]
  ([kind algorithm m]
   (->>
    (check-options m {:unsmoothed-predictions "-U"
                      :regression "-R"
                      :unpruned "-N"})
    (check-option-values m {:minimum-instances "-M"}))))

(defmethod make-classifier-options [:meta :raced-incremental-logit-boost]
  ([kind algorithm m]
   (->>
    (check-options m {:use-resampling-for-boosting "-Q"
                      :debug-mode "-D"})
    (check-option-values m {:committee-pruning-to-perform "-P"
                            :minimum-number-of-chunks "-C"
                            :name-of-base-classifier "-W"
                            :random-number-seed "-S"
                            :size-of-validation-set "-V"
                            :maximum-size-of-chunks "-M"}))))

(defmethod make-classifier-options [:meta :bagging]
  ([kind algorithm m]
   (->>
    (check-options m {:debug-mode "-D"})
    (check-option-values m {:size-of-bag "-P"
                            :parallelism "-num-slots"
                            :num-iterations "-I"
                            :random-seed "-S"
                            :name-of-base-classifier "-W"}))))

(defmethod make-classifier-options [:meta :random-subspace]
  ([kind algorithm m]
   (->>
    (check-options m {:debug-mode "-D"})
    (check-option-values m {:size-of-subspace "-P"
                            :num-iterations "-I"
                            :random-seed "-S"
                            :name-of-base-classifier "-W"}))))

(defmethod make-classifier-options [:meta :stacking]
  ([kind algorithm m]
   (->>
    (check-options m {:debug-mode "-D"})
    (check-option-values m {:cross-validation-folds "-X"
                            :meta-classifier "-M"
                            :random-seed "-S"
                            :classifiers "-B"}))))

(defmethod make-classifier-options [:meta :cost-sensitive]
  ([kind algorithm m]
   (->>
    (check-options m {:debug-mode "-D"})
    (check-option-values m {:minimize-misclassification-cost "-M"
                            :cost-file-name "-C"
                            :cost-file-directory "-N"
                            :cost-matrix "-cost-matrix"
                            :random-seed "-S"
                            :base-classifier-name "-W"}))))

(defmethod make-classifier-options [:meta :meta-cost]
  ([kind algorithm m]
   (->>
    (check-options m {:debug-mode "-D"})
    (check-option-values m {:num-bagging-iterations "-I"
                            :cost-file-name "-C"
                            :cost-file-directory "-N"
                            :cost-matrix "-cost-matrix"
                            :size-of-bag "-P"
                            :random-seed "-S"
                            :base-classifier-name "-W"}))))

(defmethod make-classifier-options [:regression :partial-least-squares]
  ([kind algorithm m]
   (->> (check-options m {:debug "-D"
                          :update-class-attribute "-U"
                          :replace-missing-values "-M"})
        (check-option-values m
                             {:algorithm "-A"
                              :type-of-preprocessing "-P"}))))

;; Building classifiers


(defn make-classifier-with
  #^{:skip-wiki true}
    [kind algorithm ^Class classifier-class options]
  (capture-out-err
   (let [options-read (if (empty? options) {} (first options))
         ^Classifier classifier (.newInstance classifier-class)
         opts (into-array String (make-classifier-options kind algorithm options-read))]
     (.setOptions classifier opts)
     classifier)))

(defmulti make-classifier
  "Creates a new classifier for the given kind algorithm and options.

   The first argument identifies the kind of classifier and the second
   argument the algorithm to use, e.g. :decision-tree :c45.

   The classifiers currently supported are:

     - :lazy :ibk
     - :decision-tree :c45
     - :decision-tree :boosted-stump
     - :decision-tree :M5P
     - :decision-tree :random-forest
     - :decision-tree :rotation-forest
     - :bayes :naive
     - :neural-network :multilayer-perceptron
     - :support-vector-machine :smo
     - :regression :linear
     - :regression :logistic
     - :regression :pace
     - :regression :pls
     - :meta :meta-cost
     - :meta :cost-sensitive
     - :meta :stacking
     - :meta :random-subspace
     - :meta :bagging
     - :meta :raced-incremental-logit-boost

   Optionally, a map of options can also be passed as an argument with
   a set of classifier specific options.

   This is the description of the supported classifiers and the accepted
   option parameters for each of them:

    * :lazy :ibk

      K-nearest neighbor classification.

      Parameters:

        - :inverse-weighted
            Neighbors will be weighted by the inverse of their distance when voting. (default equal weighting)
            Sample value: true
        - :similarity-weighted
            Neighbors will be weighted by their similarity when voting. (default equal weighting)
            Sample value: true
        - :no-normalization
            Turns off normalization.
            Sample value: true
        - :num-neighbors
            Set the number of nearest neighbors to use in prediction (default 1)
            Sample value: 3

    * :decision-tree :c45

      A classifier building a pruned or unpruned C 4.5 decision tree using
      Weka J 4.8 implementation.

      Parameters:

        - :unpruned
            Use unpruned tree. Sample value: true
        - :reduce-error-pruning
            Sample value: true
        - :only-binary-splits
            Sample value: true
        - :no-raising
            Sample value: true
        - :no-cleanup
            Sample value: true
        - :laplace-smoothing
            For predicted probabilities. Sample value: true
        - :pruning-confidence
            Threshold for pruning. Default value: 0.25
        - :minimum-instances
            Minimum number of instances per leave. Default value: 2
        - :pruning-number-folds
            Set number of folds for reduced error pruning. Default value: 3
        - :random-seed
            Seed for random data shuffling. Default value: 1

    * :bayes :naive

      Classifier based on the Bayes' theorem with strong independence assumptions, among the
      probabilistic variables.

      Parameters:

        - :kernel-estimator
            Use kernel desity estimator rather than normal. Sample value: true
        - :supervised-discretization
            Use supervised discretization to to process numeric attributes (see :supervised-discretize
            filter in clj-ml.filters/make-filter function). Sample value: true

    * :neural-network :multilayer-perceptron

      Classifier built using a feedforward artificial neural network with three or more layers
      of neurons and nonlinear activation functions. It is able to distinguish data that is not
      linearly separable.

      Parameters:

        - :no-nominal-to-binary
            A :nominal-to-binary filter will not be applied by default. (see :supervised-nominal-to-binary
            filter in clj-ml.filters/make-filter function). Default value: false
        - :no-numeric-normalization
            A numeric class will not be normalized. Default value: false
        - :no-normalization
            No attribute will be normalized. Default value: false
        - :no-reset
            Reseting the network will not be allowed. Default value: false
        - :learning-rate-decay
            Learning rate decay will occur. Default value: false
        - :learning-rate
            Learning rate for the backpropagation algorithm. Value should be between [0,1].
            Default value: 0.3
        - :momentum
            Momentum rate for the backpropagation algorithm. Value shuld be between [0,1].
            Default value: 0.2
        - :epochs
            Number of iteration to train through. Default value: 500
        - :percentage-validation-set
            Percentage size of validation set to use to terminate training. If it is not zero
            it takes precende over the number of epochs to finish training. Values should be
            between [0,100]. Default value: 0
        - :random-seed
            Value of the seed for the random generator. Values should be longs greater than
            0. Default value: 1
        - :threshold-number-errors
            The consequetive number of errors allowed for validation testing before the network
            terminates. Values should be greater thant 0. Default value: 20

    * :support-vector-machine :smo

      Support vector machine (SVM) classifier built using the sequential minimal optimization (SMO)
      training algorithm.

      Parameters:

        - :fit-logistic-models
            Fit logistic models to SVM outputs. Default value :false
        - :complexity-constant
            The complexity constance. Default value: 1
        - :tolerance
            Tolerance parameter. Default value: 1.0e-3
        - :epsilon-roundoff
            Epsilon round-off error. Default value: 1.0e-12
        - :folds-for-cross-validation
            Number of folds for the internal cross-validation. Sample value: 10
        - :random-seed
            Value of the seed for the random generator. Values should be longs greater than
            0. Default value: 1

     * :support-vector-machine :libsvm

       TODO

     * :regression :linear

      Parameters:

        - :attribute-selection
            Set the attribute selection method to use. 1 = None, 2 = Greedy. (default 0 = M5' method)
        - :keep-colinear
            Do not try to eliminate colinear attributes.
        - :ridge
            Set ridge parameter (default 1.0e-8).

     * :regression :logistic

      Parameters:

        - :max-iterations
            Set the maximum number of iterations (default -1, until convergence).
        - :ridge
            Set the ridge in the log-likelihood.

     * :meta :meta-cost

       This classifier should produce similar results to one created by passing the base learner to 
        Bagging, which is in turn passed to a CostSensitiveClassifier operating on minimum expected cost. 
        The difference is that MetaCost produces a single cost-sensitive classifier of the base learner, 
        giving the benefits of fast classification and interpretable output (if the base learner itself is interpretable).
      -from the description of MetaCost at https://weka.wikispaces.com/MetaCost

      Parameters: 

        - :num-bagging-iterations 
            Set the number of bagging iterations
        - :cost-file-name 
            Set the file path to read the cost matrix 
        - :cost-file-directory 
            Set the directory to read the cost matrix from
        - :cost-matrix 
            Specify the cost matrix as an option. 
        - :size-of-bag 
            Set the size of bag for Bagging
        - :random-seed 
            Set the random seed
        - :base-classifier-name 
            Set the (string quoted) base classifier class name 
     * :meta :cost-sensitive

      Parameters: 

  * :meta :stacking

    Combines several classifiers using the stacking method. Can do classification or regression.
    from Weka docs at http://weka.sourceforge.net/doc.dev/weka/classifiers/meta/Stacking.html.

      Parameters: 
        - :cross-validation-folds 
            Set the number of cross validation folds 
        -  :meta-classifier 
            Set the class name of the meta classifier 
        -  :random-seed 
            Set the random seed 
        -  :classifiers 
            Set the name(s) of classifiers, followed by options
  
     * :meta :random-subspace

    This method constructs a decision tree based classifier that maintains highest accuracy on 
    training data and improves on generalization accuracy as it grows in complexity. The classifier 
    consists of multiple trees constructed systematically by pseudorandomly selecting subsets of components 
    of the feature vector, that is, trees constructed in randomly chosen subspaces.
    - from Weka docs at http://weka.sourceforge.net/doc.dev/weka/classifiers/meta/RandomSubSpace.html

      Parameters: 
        - :size-of-subspace 
            Set the size of the subspace, as a function of percentage or absolute number of attributes 
        - :num-iterations 
            Set the number of iterations 
        - :random-seed 
            Set the random seed  
        - :name-of-base-classifier 
            Set the (class) name of the base classifier


     * :meta :bagging
  
    Class for bagging a classifier to reduce variance. 
    Can do classification and regression depending on the base learner. 
    - from Weka docs at http://weka.sourceforge.net/doc.dev/weka/classifiers/meta/Bagging.html

      Parameters: 
    - :num-bagging-iterations 
        Set number of bagging iterations
    -  :cost-file-name 
        Read the cost matrix from file
    -  :cost-file-directory 
        Read the cost matrix file from directory
    -  :cost-matrix 
        Read the cost matrix as an argument
    -  :size-of-bag 
        Size of each bag as a percentage of training set size
    -  :random-seed 
        Set the random seed
    -  :base-classifier-name 
        Class name of base classifier
  
  * :meta :raced-incremental-logit-boost

    Classifier for incremental learning of large datasets by way of racing logit-boosted committees.
    - from Weka docs at http://weka.sourceforge.net/doc.packages/racedIncrementalLogitBoost/weka/classifiers/meta/RacedIncrementalLogitBoost.html

      Parameters: 

    - :committee-pruning-to-perform 
        Comittee pruning to perform in the range 0 to 1.
    - :minimum-number-of-chunks
        Set the minimum chunk size
    - :name-of-base-classifier 
        Set the class name of base classifier
    - :random-number-seed 
        Set the random number seed
    - :size-of-validation-set
      Set the size of validation set
    - :maximum-size-of-chunks 
      Set the maximum number of chunks
"
  (fn [kind algorithm & options] [kind algorithm]))

(defmethod make-classifier [:lazy :ibk]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm IBk options)))

(defmethod make-classifier [:decision-tree :c45]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm J48 options)))

(defmethod make-classifier [:bayes :naive]
  ([kind algorithm & options]
   (if (or (nil? (:updateable (first options)))
           (= (:updateable (first options)) false))
     (make-classifier-with kind algorithm NaiveBayes options)
     (make-classifier-with kind algorithm NaiveBayesUpdateable options))))

(defmethod make-classifier [:neural-network :multilayer-perceptron]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm MultilayerPerceptron options)))

(defmethod make-classifier [:support-vector-machine :smo]
  ([kind algorithm & options]
   (let [options-read (if (empty? options)  {} (first options))
         classifier (new SMO)
         opts (into-array String (make-classifier-options :support-vector-machine :smo options-read))]
     (.setOptions classifier opts)
     (when (not (empty? (get options-read :kernel-function)))
         ;; We have to setup a different kernel function
       (let [kernel (get options-read :kernel-function)
             real-kernel (if (map? kernel)
                           (make-kernel-function (first (keys kernel))
                                                 (first (vals kernel)))
                           kernel)]
         (.setKernel classifier real-kernel)))
     classifier)))

(defmethod make-classifier [:support-vector-machine :spegasos]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm SPegasos options)))

(defmethod make-classifier [:support-vector-machine :libsvm]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm LibSVM options)))

(defmethod make-classifier [:support-vector-machine :libsvm-grid]
  ([kind algorithm & options]
   (for [c (range -5 17 2) g (range 3 -17 -2)]
     (make-classifier-with
      :support-vector-machine :libsvm
      LibSVM (concat options [:param-C (Math/pow 2.0 c)
                              :kernel-gamma (Math/pow 2.0 g)])))))

(defmethod make-classifier [:regression :linear]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm LinearRegression options)))

(defmethod make-classifier [:regression :logistic]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm Logistic options)))

(defmethod make-classifier [:regression :pace]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm PaceRegression options)))

(defmethod make-classifier [:regression :boosted-regression]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm AdditiveRegression options)))

(defmethod make-classifier [:decision-tree :boosted-stump]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm LogitBoost options)))

(defmethod make-classifier [:decision-tree :random-forest]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm RandomForest options)))

(defmethod make-classifier [:decision-tree :rotation-forest]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm RotationForest options)))

(defmethod make-classifier [:decision-tree :m5p]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm M5P options)))

(defmethod make-classifier [:meta :raced-incremental-logit-boost]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm RacedIncrementalLogitBoost options)))

(defmethod make-classifier [:meta :bagging]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm Bagging options)))

(defmethod make-classifier [:meta :random-subspace]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm RandomSubSpace options)))

(defmethod make-classifier [:meta :stacking]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm Stacking options)))

(defmethod make-classifier [:meta :cost-sensitive]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm CostSensitiveClassifier options)))

(defmethod make-classifier [:meta :meta-cost]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm MetaCost options)))

(defmethod make-classifier [:regression :partial-least-squares]
  ([kind algorithm & options]
   (make-classifier-with kind algorithm PLSClassifier options)))

;; Training classifiers

(defn classifier-train
  "Trains a classifier with the given dataset as the training data."
  ([^Classifier classifier dataset]
   (do (.buildClassifier classifier dataset)
       classifier)))

(defn classifier-copy
  "Performs a deep copy of the classifier"
  [^Classifier classifier]
  (AbstractClassifier/makeCopy classifier))

(defn classifier-copy-and-train
  "Performs a deep copy of the classifier, trains the copy, and returns it."
  [classifier dataset]
  (classifier-train (classifier-copy classifier) dataset))

(defn classifier-update
  "If the classifier is updateable it updates the classifier with the given instance or set of instances."
  ([^Classifier classifier instance-s]
     ;; Arg... weka doesn't provide a formal interface for updateClassifer- How do I type hint this?
   (if (is-dataset? instance-s)
     (do (doseq [i (dataset-seq instance-s)]
           (.updateClassifier classifier ^Instance i))
         classifier)
     (do (.updateClassifier classifier ^Instance instance-s)
         classifier))))

;; Evaluating classifiers

(defn- collect-evaluation-results
  "Collects all the statistics from the evaluation of a classifier."
  ([class-labels ^Evaluation evaluation]
   {:confusion-matrix (try (.toMatrixString evaluation) (catch Exception e nil))
    :summary (.toSummaryString evaluation)
    :correct (try-metric #(.correct evaluation))
    :incorrect (try-metric #(.incorrect evaluation))
    :unclassified (try-metric #(.unclassified evaluation))
    :percentage-correct (try-metric #(.pctCorrect evaluation))
    :percentage-incorrect (try-metric #(.pctIncorrect evaluation))
    :percentage-unclassified (try-metric #(.pctUnclassified evaluation))
    :error-rate (try-metric #(.errorRate evaluation))
    :mean-absolute-error (try-metric #(.meanAbsoluteError evaluation))
    :relative-absolute-error (try-metric #(.relativeAbsoluteError evaluation))
    :root-mean-squared-error (try-metric #(.rootMeanSquaredError evaluation))
    :root-relative-squared-error (try-metric #(.rootRelativeSquaredError evaluation))
    :correlation-coefficient (try-metric #(.correlationCoefficient evaluation))
    :average-cost (try-metric #(.avgCost evaluation))
    :kappa (try-metric #(.kappa evaluation))
    :kb-information (try-metric #(.KBInformation evaluation))
    :kb-mean-information (try-metric #(.KBMeanInformation evaluation))
    :kb-relative-information (try-metric #(.KBRelativeInformation evaluation))
    :sf-entropy-gain (try-metric #(.SFEntropyGain evaluation))
    :sf-mean-entropy-gain (try-metric #(.SFMeanEntropyGain evaluation))
    :roc-area (try-multiple-values-metric class-labels (fn [i] (try-metric #(.areaUnderROC evaluation i))))
    :false-positive-rate (try-multiple-values-metric class-labels (fn [i] (try-metric #(.falsePositiveRate evaluation i))))
    :false-negative-rate (try-multiple-values-metric class-labels (fn [i] (try-metric #(.falseNegativeRate evaluation i))))
    :f-measure (try-multiple-values-metric class-labels (fn [i] (try-metric #(.fMeasure evaluation i))))
    :precision (try-multiple-values-metric class-labels (fn [i] (try-metric #(.precision evaluation i))))
    :recall (try-multiple-values-metric class-labels (fn [i] (try-metric #(.recall evaluation i))))
    :evaluation-object evaluation}))

(defmulti classifier-evaluate
  "Evaluates a trained classifier using the provided dataset or cross-validation.
   The first argument must be the classifier to evaluate, the second argument is
   the kind of evaluation to do.
   Two possible evaluations ara availabe: dataset and cross-validations. The values
   for the second argument can be:

    - :dataset
    - :cross-validation

    * :dataset

    If dataset evaluation is desired, the function call must receive as the second
    parameter the keyword :dataset and as third and fourth parameters the original
    dataset used to build the classifier and the training data:

      (classifier-evaluate *classifier* :dataset *training* *evaluation*)

    * :cross-validation

    If cross-validation is desired, the function call must receive as the second
    parameter the keyword :cross-validation and as third and fourth parameters the dataset
    where for training and the number of folds.

      (classifier-evaluate *classifier* :cross-validation *training* 10)
    
    An optional seed can be provided for generation of the cross validation folds.

      (classifier-evaluate *classifier* :cross-validation *training* 10 {:random-seed 29})

    The metrics available in the evaluation are listed below:

    - :correct
        Number of instances correctly classified
    - :incorrect
        Number of instances incorrectly evaluated
    - :unclassified
        Number of instances incorrectly classified
    - :percentage-correct
        Percentage of correctly classified instances
    - :percentage-incorrect
        Percentage of incorrectly classified instances
    - :percentage-unclassified
        Percentage of not classified instances
    - :error-rate
    - :mean-absolute-error
    - :relative-absolute-error
    - :root-mean-squared-error
    - :root-relative-squared-error
    - :correlation-coefficient
    - :average-cost
    - :kappa
        The kappa statistic
    - :kb-information
    - :kb-mean-information
    - :kb-relative-information
    - :sf-entropy-gain
    - :sf-mean-entropy-gain
    - :roc-area
    - :false-positive-rate
    - :false-negative-rate
    - :f-measure
    - :precision
    - :recall
    - :evaluation-object
        The underlying Weka's Java object containing the evaluation
"
  (fn [classifier mode & evaluation-data] mode))

(defmethod classifier-evaluate :dataset
  ([^Classifier classifier mode & [training-data test-data]]
   (capture-out-err
    (letfn [(eval-fn [c]
              (let [evaluation (new Evaluation training-data)
                    class-labels (dataset-class-labels training-data)]
                (.evaluateModel evaluation c test-data (into-array []))
                (collect-evaluation-results class-labels evaluation)))]
      (if (seq? classifier)
        (last (sort-by :correct (map eval-fn classifier)))
        (eval-fn classifier))))))

(defmethod classifier-evaluate :cross-validation
  ([classifier mode & [training-data folds
                       {:keys [random-seed]
                        :or {random-seed (.getTime (new Date))}}
                       ]]
   (capture-out-err
    (letfn [(eval-fn [c]
              (let [evaluation (new Evaluation training-data)
                    class-labels (dataset-class-labels training-data)]
                (.crossValidateModel evaluation c training-data folds
                                     (new Random random-seed) (into-array []))
                (collect-evaluation-results class-labels evaluation)))]
      (if (seq? classifier)
        (last (sort-by :correct (map eval-fn classifier)))
        (eval-fn classifier))))))

;; Classifying instances

(defn classifier-classify
  "Classifies an instance using the provided classifier. Returns the
   class as a keyword."
  ([^Classifier classifier ^Instance instance]
   (let [pred (.classifyInstance classifier instance)]
     (keyword (.value (.classAttribute instance) pred)))))

(defn classifier-predict-numeric
  "Predicts the class attribute of an instance using the provided
   classifier. Returns the value as a floating-point value (e.g., for
   regression)."
  ([^Classifier classifier ^Instance instance]
   (.classifyInstance classifier instance)))

(defn classifier-label
  "Classifies and assign a label to a dataset instance.
   The function returns the newly classified instance. This call is
   destructive, the instance passed as an argument is modified."
  ([^Classifier classifier ^Instance instance]
   (let [cls (.classifyInstance classifier instance)]
     (doto instance (.setClassValue cls)))))

(defn classifier-predict-probability
  "Classifies an instance using the provided classifier. Returns the
   probability distribution across classes for the instance"
  ([^Classifier classifier ^Instance instance]
   (vec (.distributionForInstance classifier instance))))
