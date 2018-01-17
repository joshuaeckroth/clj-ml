(ns clj-ml.classifiers-test
  (:use [clj-ml classifiers data] :reload-all)
  (:use clojure.test midje.sweet))

(deftest make-classifiers-options-ibk
  (fact
   (let [options (make-classifier-options
                  :lazy :ibk
                  {:inverse-weighted true :similarity-weighted true :no-normalization true :num-neighbors 3})]
     options => (just ["-I" "-F" "-N" "-K" "3"] :in-any-order))))

(deftest make-classifier-ibk
  (let [c (make-classifier :lazy :ibk)]
    (is (= (class c)
           weka.classifiers.lazy.IBk))))

(deftest train-classifier-ibk
  (let [c (make-classifier :lazy :ibk)
        ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])]
    (dataset-set-class ds 2)
    (classifier-train c ds)
    (is true)))

(deftest make-classifiers-options-c45
  (fact
   (let [options (make-classifier-options
                  :decision-tree :c45
                  {:unpruned true :reduced-error-pruning true :only-binary-splits true :no-raising true
                   :no-cleanup true :laplace-smoothing true :pruning-confidence 0.12 :minimum-instances 10
                   :pruning-number-folds 5 :random-seed 1})]
     options => (just ["-U" "-R" "-B" "-S" "-L" "-A" "-C" "0.12" "-M" "10" "-N" "5" "-Q" "1"] :in-any-order))))

(deftest make-classifier-c45
  (let [c (make-classifier :decision-tree :c45)]
    (is (= (class c)
           weka.classifiers.trees.J48))))

(deftest train-classifier-c45
  (let [c (make-classifier :decision-tree :c45)
        ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])]
    (dataset-set-class ds 2)
    (classifier-train c ds)
    (is true)))

(deftest make-classifier-bayes
  (fact
   (let [c (make-classifier :bayes :naive {:kernel-estimator true :old-format true})
         opts (vec (.getOptions c))]
     opts => (contains ["-K" "-O"]))))

(deftest make-classifier-bayes-updateable
  (let [c (make-classifier :bayes :naive {:updateable true})]
    (is (= (class c)
           weka.classifiers.bayes.NaiveBayesUpdateable))))

(deftest train-classifier-bayes
  (let [c (make-classifier :bayes :naive {:kernel-estimator true :old-format true})
        ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])]
    (dataset-set-class ds 2)
    (classifier-train c ds)
    (is true)))

(deftest classifier-evaluate-dataset
  (let [c   (make-classifier :decision-tree :c45)
        ds  (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        tds (make-dataset "test" [:a :b {:c [:m :n]}] [[4 1 :n] [4 5 :m]])
        _   (dataset-set-class ds 2)
        _   (dataset-set-class tds 2)
        _   (classifier-train c ds)
        res (classifier-evaluate c :dataset ds tds)]
    (is (= 28 (count (keys res))))))

(deftest make-classifier-svm-smo-polykernel
  (let [svm (make-classifier :support-vector-machine :smo {:kernel-function {:polynomic {:exponent 2.0}}})]
    (is (= weka.classifiers.functions.supportVector.PolyKernel
           (class (.getKernel svm))))))

(deftest classifier-evaluate-cross-validation
  (let [c (make-classifier :decision-tree :c45)
        ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        _  (dataset-set-class ds 2)
        _  (classifier-train c ds)
        res (classifier-evaluate c :cross-validation ds 2)
        randres (classifier-evaluate c :cross-validation ds 2 {:random-seed 29}) ]
    (is (= 28 (count (keys res))))
    (is (= 28 (count (keys randres))))))

(deftest classifier-evaluate-cross-validation-grid
  (let [c (make-classifier :support-vector-machine :libsvm-grid)
        ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        _  (dataset-set-class ds 2)
        res (classifier-evaluate c :cross-validation ds 2)]
    (is (= 28 (count (keys res))))))

(deftest test-classifier-classify
  (let [c (make-classifier :decision-tree :c45)
        ds (-> (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
               (dataset-set-class 2))
        inst (-> (first (dataset-seq ds))
                 (instance-set-class-missing))]
    (classifier-train c ds)
    (is (= :m (classifier-classify c inst)))))

(deftest test-classifier-predict-probability
  (let [c (make-classifier :decision-tree :c45)
        ds (-> (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
               (dataset-set-class 2))
        inst (-> (first (dataset-seq ds))
                 (instance-set-class-missing))
        _ (classifier-train c ds)
        res (classifier-predict-probability c inst)]
    (is (= 2 (count res)))
    (is (every? number? res))))

(deftest test-classifier-label
  (let [c (make-classifier :decision-tree :c45)
        ds (-> (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
               (dataset-set-class 2))
        inst (-> (first (dataset-seq ds))
                 (instance-set-class-missing))]
    (is (= nil (instance-get-class inst)))
    (classifier-train c ds)
    (classifier-label c inst)
    (is (= :m (instance-get-class inst)))))

(deftest update-updateable-classifier
  (let [c (make-classifier :bayes :naive {:updateable true})
        ds (make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        _  (dataset-set-class ds 2)
        inst (make-instance ds {:a 56 :b 45 :c :m})]
    (classifier-train  c ds)
    (classifier-update c ds)
    (classifier-update c inst)
    (is true)))

(deftest make-classifiers-options-raced-incremental-logit-boost
  (fact
   (let [options (make-classifier-options
                  :meta :raced-incremental-logit-boost
                  {:use-resampling-for-boosting true :debug-mode true
                   :committee-pruning-to-perform 0 :minimum-number-of-chunks 200
                   :name-of-base-classifier "weka.classifiers.trees.DecisionStump"
                   :random-number-seed 20 :size-of-validation-set 200
                   :maximum-size-of-chunks 400})]
     options => (just ["-Q" "-D" "-P" "0" "-C" "200" "-W" "weka.classifiers.trees.DecisionStump" "-S" "20" "-V" "200" "-M" "400"] :in-any-order))))

(deftest make-classifier-raced-incremental-logit-boost
  (let [c (make-classifier :meta :raced-incremental-logit-boost)]
    (is (= (class c)
           weka.classifiers.meta.RacedIncrementalLogitBoost))))

(deftest update-updateable-classifier-logit-boost
  (let [c (clj-ml.classifiers/make-classifier :meta :raced-incremental-logit-boost)
        ds (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        _  (dataset-set-class ds 2)
        inst (make-instance ds {:a 56 :b 45 :c :m})]
    (classifier-train  c ds)
    (classifier-update c ds)
    (classifier-update c inst)
    (is true)))

(deftest make-classifiers-options-pls
  (fact
   (let [options (make-classifier-options
                  :regression :partial-least-squares
                  {:algorithm "PLS1" :type-of-preprocessing "standardize"})]
     options => (just ["-A" "PLS1" "-P" "standardize"] :in-any-order))))

(deftest make-classifier-pls
  (let [c (make-classifier :regression :partial-least-squares)]
    (is (= (class c)
           weka.classifiers.functions.PLSClassifier))))

(deftest make-classifier-random-forest
  (let [c (make-classifier :decision-tree :random-forest
                           {:num-trees-in-forest 10
                            :parallelism 2}
                           )]
    (is (= (class c)
           weka.classifiers.trees.RandomForest))))

(deftest make-classifier-bagging
  (let [c (make-classifier :meta :bagging
                           {:size-of-bag 10
                            :parallelism 2}
                           )]
    (is (= (class c)
           weka.classifiers.meta.Bagging))))

(deftest make-classifier-random-subspace
  (let [c (make-classifier :meta :random-subspace
                           {:size-of-subspace 10
                            :num-iterations 20}
                           )]
    (is (= (class c)
           weka.classifiers.meta.RandomSubSpace))))

(deftest make-classifier-stacking
  (let [c (make-classifier :meta :stacking
                           {:cross-validation-folds 3 
                            :random-seed 20}
                           )]
    (is (= (class c)
           weka.classifiers.meta.Stacking))))            

(deftest make-classifier-cost-sensitive
  (let [c (make-classifier :meta :cost-sensitive
                           {:random-seed 20
                            :base-classifier-name "weka.classifiers.trees.DecisionStump"
                            :cost-matrix "[0.0 5.0; 1.0 0.0]"
                            }
                           )]
    (is (= (class c)
           weka.classifiers.meta.CostSensitiveClassifier))))         

(deftest make-classifier-meta-cost
  (let [c (make-classifier :meta :meta-cost
                           {:random-seed 20
                            :base-classifier-name "weka.classifiers.trees.DecisionStump"
                            :cost-matrix "[0.0 5.0; 1.0 0.0]"
                            }
                           )]
    (is (= (class c)
           weka.classifiers.meta.MetaCost))))

(deftest make-classifier-boosted-stump
  (let [c (make-classifier :decision-tree :boosted-stump
                           {:weak-learning-class "weka.classifiers.trees.DecisionStump"
                              :num-iterations 10 
                              :random-seed 29 
                              :percentage-weight-mass 90 
                              :log-likelihood-improvement-threshold 0.9 
                              :z-max-threshold-for-responses 3 
                              :shrinkage-parameter 1}
                           )]
    (is (= (class c)
           weka.classifiers.meta.LogitBoost))))
