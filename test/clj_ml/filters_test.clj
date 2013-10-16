(ns clj-ml.filters-test
  (:use [clj-ml filters data io] :reload-all)
  (:use clojure.test midje.sweet))

(deftest make-filter-options-supervised-discretize
  (fact
   (let [options (make-filter-options :supervised-discretize {:attributes [1 2] :invert true :binary true :better-encoding true :kononenko true :nonexitent true})]
     options => (just ["-R" "2,3" "-V" "-D" "-E" "-K"] :in-any-order))))

(deftest make-filter-options-unsupervised-discretize
  (fact
   (let [options (make-filter-options :unsupervised-discretize {:attributes [1 2] :binary true
                                                                :better-encoding true :equal-frequency true :optimize true
                                                                :number-bins 4 :weight-bins 1})]
     options => (just ["-R" "2,3" "-D" "-E" "-F" "-O" "-B" "4" "-M" "1"] :in-any-order))))

(deftest make-filter-options-supervised-nominal-to-binary
  (fact
   (let [options (make-filter-options :supervised-nominal-to-binary {:also-binary true :for-each-nominal true})]
     options => (just ["-N" "-A"] :in-any-order))))

(deftest make-filter-options-unsupervised-nominal-to-binary
  (fact
   (let [options (make-filter-options :unsupervised-nominal-to-binary {:attributes [1,2] :also-binary true :for-each-nominal true :invert true})]
     options => (just ["-R" "2,3" "-V" "-N" "-A"] :in-any-order))))

(deftest make-filter-options-string-to-word-vector
  (fact
   (let [options (make-filter-options :string-to-word-vector
                                      {:attributes [1] :lowercase true :counts false
                                       :words-to-keep 20 :transform-idf true
                                       :stemmer "weka.core.stemmers.SnowballStemmer -S English"})]
     options => (just ["-R" "2" "-L" "-W" "20" "-I" "-stemmer" "weka.core.stemmers.SnowballStemmer -S English"] :in-any-order))))

(deftest make-filter-remove-useless-attributes
  (let [ds (make-dataset :foo [:a] [[1] [2]])
        filter (make-filter :remove-useless-attributes {:dataset-format ds :max-variance 95})]
    (is (== (.getMaximumVariancePercentageAllowed filter) 95))))

(deftest make-filter-resample-unsupervised
  (fact
   (let [ds (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
                (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
         options (make-filter-options :resample-unsupervised
                                      {:dataset-format ds :seed 10 :size-percent 50 :no-replacement true :invert true})]
     options => (just ["-S" "10" "-Z" "50" "-V" "-no-replacement"] :in-any-order))))

(deftest make-filter-resample-supervised
  (fact
   (let [ds (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
                (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
         options (make-filter-options :resample-supervised
                                      {:dataset-format ds :seed 10 :size-percent 50 :no-replacement true :invert true :bias 1})]
     options => (just ["-S" "10" "-Z" "50" "-V" "-no-replacement" "-B" "1"] :in-any-order))))

(deftest make-filter-discretize-sup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        _ (dataset-set-class ds 2)
        f (make-filter :supervised-discretize {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.supervised.attribute.Discretize
           (class f)))))

(deftest make-filter-discretize-unsup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        f (make-filter :unsupervised-discretize {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Discretize
           (class f)))))

(deftest make-filter-nominal-to-binary-sup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        foo1(dataset-set-class ds 2)
        f (make-filter :supervised-nominal-to-binary {:dataset-format ds})]
    (is (= weka.filters.supervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-nominal-to-binary-unsup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        f (make-filter :unsupervised-nominal-to-binary {:dataset-format ds :attributes [2]})]
    (is (= weka.filters.unsupervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-string-to-word-vector
  (let [ds (make-dataset :test [{:s nil} {:class [:yes :no]}]
                         [["Hello, world!" :no] ["This is a test, is a world." :yes]])
        f (make-filter :string-to-word-vector {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.StringToWordVector
           (class f)))))

(deftest make-filter-reorder-attributes
  (let [ds (make-dataset :test [{:class [:yes :no]} {:s nil} :n]
                         [[:yes "Hello" 55] [:no "World" -100]])
        f (make-filter :reorder-attributes {:dataset-format ds :attributes ["2-last" "1"]})]
    (is (= weka.filters.unsupervised.attribute.Reorder
           (class f)))))

(deftest make-filter-resample-unsupervised
  (let [ds (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
               (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
        f (make-filter :resample-unsupervised {:dataset-format ds :seed 10 :size-percent 50 :no-replacement true})]
    (is (= weka.filters.unsupervised.instance.Resample
           (class f)))))

(deftest make-filter-resample-supervised
  (let [ds (dataset-set-class
            (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
                (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
            :class)
        f (make-filter :resample-supervised {:dataset-format ds :seed 10 :size-percent 50 :no-replacement true :bias 1})]
    (is (= weka.filters.supervised.instance.Resample
           (class f)))))

(deftest make-filter-remove-attributes
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        f (make-filter :remove-attributes {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Remove
           (class f)))
    (let [res (filter-apply f ds)]
      (is (= (dataset-format res)
             [:b {:c '(:g :m)}])))))

(deftest make-apply-filter-remove-attributes
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        res (make-apply-filter :remove-attributes {:attributes [0]} ds)]
    (is (= (dataset-format res)
           [:b {:c '(:g :m)}]))))

(deftest remove-precentage-test
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 2 :m]
                           [4 5 :g]])]
    (is (= (dataset-count (remove-percentage ds {:percentage 75})) 1))))

(deftest remove-range-test
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 2 :m]
                           [4 5 :g]])]
    (is (= (dataset-count (remove-range ds {:range "first-3"})) 1)
        (= (dataset-count (remove-range ds {:range "first-3" :invert true})) 3))))

(deftest make-apply-filter-numeric-to-nominal
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])]
    (testing "when no attributes are specified"
      (is (= (dataset-format  (make-apply-filter :numeric-to-nominal {} ds))
             [{:a '(:1 :2 :4)} {:b '(:2 :3 :5)} {:c '(:g :m)}])))
    (testing "when attributes are specified by index"
      (is (= (dataset-format  (make-apply-filter :numeric-to-nominal {:attributes [0]} ds))
             [{:a '(:1 :2 :4)} :b {:c '(:g :m)}])))
    (testing "when attributes are specified by name"
      (is (= (dataset-format  (make-apply-filter :numeric-to-nominal {:attributes [:b]} ds))
             [:a {:b '(:2 :3 :5)} {:c '(:g :m)}])))))

(deftest make-apply-filter-string-to-word-vector
  (let [ds (make-dataset :test [{:s nil} {:class [:yes :no]}]
                         [["Hello, world! tests Dogs cats" :no]
                          ["This is a test, is a world." :yes]])]
    (is (= (map instance-to-map
                (dataset-seq (make-apply-filter :string-to-word-vector
                                                {:attributes [0] :counts true :lowercase true
                                                 :stemmer "weka.core.stemmers.SnowballStemmer -S English"}
                                                ds)))
           '({:world 1.0, :this 0.0, :test 1.0, :is 0.0, :hello 1.0,
              :dog 1.0, :cat 1.0, :a 0.0, :class :no}
             {:world 1.0, :this 1.0, :test 1.0, :is 2.0, :hello 0.0,
              :dog 0.0, :cat 0.0, :a 2.0, :class :yes})))))

(deftest make-apply-filter-add-attribute
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        res (add-attribute ds {:type :nominal, :column 1, :name "pet", :labels ["dog" "cat"]})]
    (is (= (dataset-format res)
           [:a {:pet '(:dog :cat)} :b {:c '(:g :m)}]))))

(deftest make-apply-filter-reorder-attributes
  (let [ds (make-dataset :test [{:class [:yes :no]} {:s nil} :n]
                         [[:yes "Hello" 55] [:no "World" -100]]) 
        ds2 (make-apply-filter :reorder-attributes {:attributes ["2-last" "1"]} ds)]
    (is (= (str ds (str (make-dataset :test [{:s nil} :n {:class [:yes :no]}]
                                      [["Hello" 55 :yes] ["World" -100 :no]])))))))

(deftest make-apply-filter-resample-unsupervised
  (let [ds (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
               (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
        ds2 (make-apply-filter :resample-unsupervised {:seed 10 :size-percent 50 :no-replacement true} ds)]
    (is (= 75 (dataset-count ds2)))))

(deftest make-apply-filter-resample-supervised
  (let [ds (dataset-set-class
            (do (println "Loading instances from http://clj-ml.artifice.cc/iris.arff ...")
                (load-instances :arff "http://clj-ml.artifice.cc/iris.arff"))
            :class)
        ds2 (make-apply-filter :resample-supervised {:seed 10 :size-percent 50 :no-replacement true :bias 1} ds)]
    (is (= 75 (dataset-count ds2)))))

(deftest make-apply-filters-test
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        res (make-apply-filters
             [[:add-attribute {:type :nominal, :column 1, :name "pet", :labels ["dog" "cat"]}]
              [:remove-attributes {:attributes [:a :c]}]] ds)]
    (is (= (dataset-format res)
           [{:pet '(:dog :cat)} :b]))))

(deftest using-regular-filter-fns-with-threading
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        res (-> ds
                (add-attribute {:type :nominal, :column 1, :name "pet", :labels ["dog" "cat"]})
                (remove-attributes {:attributes [:a :c]}))]
    (is (= (dataset-format res)
           [{:pet '(:dog :cat)} :b]))))

(deftest make-apply-filter-clj-streamable
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])

        rename-attributes (fn [^weka.core.Instances input-format]
                            (doto (weka.core.Instances. input-format 0)
                              (.renameAttribute 0 "foo")
                              (.renameAttribute 1 "bar")))
        inc-nums (fn [^weka.core.Instance instance]
                   (doto (.copy instance)
                     (.setValue 0 (inc (.value instance 0)))
                     (.setValue 1 (+ (.value instance 0) (.value instance 1)))))
        res (make-apply-filter :clj-streamable
                               {:process inc-nums
                                :determine-dataset-format rename-attributes} ds)]
    (is (= (map instance-to-map (dataset-seq res))
           [{:foo 2.0 :bar 3.0 :c :g}
            {:foo 3.0 :bar 5.0 :c :m}
            {:foo 5.0 :bar 9.0 :c :g}]))))


(deftest make-apply-filter-clj-batch
  (let [ds (make-dataset :test [:a]
                         [ [1]
                           [2]
                           [4]])
        max-diff-attr (weka.core.Attribute. "max-diff")
        add-max-diff-attr (fn [^weka.core.Instances input-format]
                            (doto (weka.core.Instances. input-format 0)
                              (.insertAttributeAt max-diff-attr 1)))
        add-max-diff-values (fn [^weka.core.Instances instances]
                              (let [ds-seq (dataset-seq instances)
                                    a-max (apply max (map #(.value % 0) ds-seq))
                                    result (add-max-diff-attr instances)
                                    add-instance #(.add result %)]
                                (doseq [instance ds-seq]
                                  (-> instance
                                      instance-to-vector
                                      (conj (- a-max (.value instance 0)))
                                      (#(weka.core.DenseInstance. 1 (into-array Double/TYPE %)))
                                      add-instance))
                                result))
        res (clj-batch ds
                       {:process add-max-diff-values
                        :determine-dataset-format add-max-diff-attr})]
    (is (= [{:a 1.0 :max-diff 3.0}
            {:a 2.0 :max-diff 2.0}
            {:a 4.0 :max-diff 0.0}]
           (dataset-as-maps res)))))
