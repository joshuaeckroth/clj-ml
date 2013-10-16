(defproject cc.artifice/clj-ml "0.6.0-SNAPSHOT"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :java-source-paths ["src/java"]
  :license {:name "MIT License"
            :url "http://opensource.org/licenses/MIT"}
  :url "https://github.com/joshuaeckroth/clj-ml"
  :dependencies [[nz.ac.waikato.cms.weka/weka-dev "3.7.10"]
                 [org.clojure/data.xml "0.0.7"]
                 [org.apache.lucene/lucene-analyzers-common "4.3.0"]]
  :profiles {:dev
             {:plugins [[lein-midje "2.0.0"]]
              :dependencies [[midje "1.4.0"]]}}
  :codox {:output-dir "website/doc"
          :src-dir-uri "http://github.com/joshuaeckroth/clj-ml/blob/master"
          :src-linenum-anchor-prefix "L"})
