(ns nlp-demo.core
  (:require [clojure.pprint :refer [pprint]]
            [opennlp.nlp :as nlp]
            [opennlp.treebank :as tb]
            [opennlp.tools.train :as train]))

;; splitting into sentences
(def get-sentences (nlp/make-sentence-detector "models/en-sent.bin"))
(get-sentences "First sentence. Second sentence? Here is another one. And so on and so forth - you get the idea.... And other stuff")

;; tokenizing
(def tokenize (nlp/make-tokenizer "models/en-token.bin"))
(tokenize "No person can cross the same river twice, because neither the person nor the river are the same.")











;; part-of-speech (POS) tagging
;; https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
;; PRP - Personal pronoun
;; VBZ - Verb, 3rd person singular present
;; IN - Preposition or subordinating conjunction
;; DT - Determiner
;; TO - to
;; VB - Verb, base form
(def pos-tag (nlp/make-pos-tagger "models/en-pos-maxent.bin"))
(pprint (pos-tag (tokenize "The dog jumped over the moon.")))
(pprint (pos-tag (tokenize "It's that that needs to change.")))
(pprint (meta (pos-tag (tokenize "It's that that needs to change."))))

;; chunking
(def chunker (tb/make-treebank-chunker "models/en-chunker.bin"))
(pprint (chunker (pos-tag (tokenize "Peter Piper picked a peck of pickled peppers."))))







;; using existing named entity recognition (NER) models
;; http://opennlp.sourceforge.net/models-1.5/
(def name-find (nlp/make-name-finder "models/en-ner-person.bin"))
(name-find (tokenize "What is Grace Hopper's favorite algorithm?"))
;; watch out, en-ner-person.bin probably uses names that are common in english, not Dutch
(name-find (tokenize "What is Edsger Dijkstra's favorite algorithm?"))

;; training an NER model
(def find-predator (nlp/make-name-finder (train/train-name-finder "training/predator.train")))
(def s "The cat tries different plans to catch the bird.")
;; Oops, we tagged noun phrases, not just nouns so this doesn't work
(find-predator (tokenize s))
(find-predator (tb/phrase-strings (chunker (pos-tag (tokenize s)))))










;; training a categorization model
(def calendar-intent (nlp/make-document-categorizer (train/train-document-categorization "training/calendar.train")))
(calendar-intent "When's the next time I meet up with Jenny?")
(meta (calendar-intent "Buffalo buffalo."))










;; making a human language query system
;; train a model to extract attribute names
(def find-attribute (nlp/make-name-finder (train/train-name-finder "training/attribute.train")))

(defn get-attribute-entities [s]
  (let [[[name] [attribute]] (-> s
                                 tokenize
                                 ((juxt name-find find-attribute)))]
    {:name name :attribute attribute}))

(get-attribute-entities "What is Joe's hair color?")
(get-attribute-entities "When was Jill born?")
(get-attribute-entities "What's Matt Johnson's birth date?")


(def people-db-intent (nlp/make-document-categorizer (train/train-document-categorization "training/people-db.train")))
(people-db-intent "What's Brian's hometown?")
(get-attribute-entities  "What's Brian's hometown?")
