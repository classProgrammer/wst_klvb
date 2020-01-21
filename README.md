# Information Retrieval
## Buzzword Bingo
- Information Retrieval (IR)
  - finding material (usually
documents) of an unstructured nature (usually text)
that satisfies an information need of users from within
large collections of documents (usually stored on
computers).
- IR Process
  - Data -> Create Document Representation -> Index Docs -> Ready
    - Done beforehand
  - User query -> create query representation -> Search Indexed Docs -> Order
- Retrieval Model
  - Quality of a retrieval model depends on how well it
matches user needs !
  - Determines
    - the structure of the document representation
    - the structure of the query representation
    - the similarity matching function
- Query
  - what users look for
  - list of words or phrases
- Document
  - Data
  - Bag of words
- Corpus/Collection
  - Set of documents
- Index
  - representation of information to make querying easier
  - Manual Indexing
    - Pro: Human indexers can establish relationships and
concepts between seemingly different topics that
can be very useful to future readers
    - Con: Slow and expensive 
  - Automatic Indexing
    - Inverted Index
      - map terms to docs
      - finds docs matching query
      - ![](img/invert_idx.PNG)
    - Indexing with Controlled Vocabulary
      - Find synonyms
      - describe concepts with lists of terms
      - e.g. Vehicle, Car, Mercedes, Taxi all describe the same concept
      - can increase performance 
    - Vector Space Models
      - map docs to terms
      - find docs matching query
      - ranks docs for best match
- Controlled Vocabulary
  - relationship between terms
  - find synonyms
  - List
    - e.g. alphabetical list
  - Ring
    - set of X terms
    - used in search engines
  - Taxonomy
    - hiarchical classification system
    - each term has one or more broader terms except the top term
    - each term has one or more narrower terms except the bottom terms
  - Thesaurus
    - taxonomy + additional relationships
- Boolean Queries
  - OR
  - AND
  - BUT
  - Ranking hard => yes, no answer
  - Relevance feedback hard => no ranking
  - all matched docs are returned
  - complex requests hard to write
- Union
  - everything in A + everything in B = OR
- Intersection
  - What's in both = AND
  - performance -> start with smallest set then keep ANDing
- Difference
  - (A OR B) - (A AND B) = BUT 
- Term
  - word/concept in document or query
  - weighting 
    - too frequent
    - significant
    - too rare
    - document with 10 occurences of a term is more important than a document with 1 occurance BUT not 10 times as important => weighting becasue relevance doesn't increase proportionally
- Dictionary
  - sorted list of terms used by the index
- Document Processing Pipeline / Normalization
  - Text
  - remove properties and formatting
  - parse
  - remove stopwords
  - stemming or lemmatization
  - synonym matching
  - indexing
- Token
  - small unit of meaningful text
- Tokenization
  - break into tokens on whitespaces
- Tokenization Problems
  - Specific domains like biomedical texts have lots of unusual symbols/special terms that should be interpreted correctly
  - Semantic meaning could be lost
- Lemmatization/Stemming
  - Stem = cut off
    - walk, walked, walking => walk => walk(ed | ing)
  - Lemmatizaton = get words like in dictionary through morphological analysis
- Morphology
  - knowledge how words are morphed => write, wrote, written = write
- Stop Word removal
  - Stop Word
    - small/no semantic content
    - the, a, an, is
- Ranked Retrieval
  - more relevant = higher up
  - Jaccard Coefficinet
    - CONS:
      - term frequency doesn't matter
      - but rare terms are more informative than frequent ones
  - Statistical Models
    - vector space model
    - statistical info used for ranking => term frequency
    - Ranked based on similarity to query
    - similarity based on frequency of keywords
- Vector Space Model
  - Docs and queries are represente as N-dimensional vectors
  - Terms get a weight
  - Terms = Axes of the Vector Space
  - Documents are points or vectors in the vector space
  - Document collection can be represented as term-document matrix
    - Entry = Weight of term in a document
    - Term Frequnecy = frequenzy / most commont term
  - Document frequency
    - number of documents containing the term
  - Inverse Document Frequency
    - Terms that appear in many different documents are less
  - CONS
    - missing semantic info
    - missing syntactic info
    - assumption of term independenc (ignores synonyms)
    - Lacks the control of a Boolean model (e.g.,
indicative of overall topic
  - Term Frequency + Inverse Document Frequency:
    - TF-IDF Weighting
    - most common term weighting approach (vector-space model)
    - A term occurring frequently in the document but rarely in the rest of the collection is given high weight
- Similarity
  - Euclidean distance
  - Vector Product
  - Cosine similarity
  
# Clustering Classification
## Buzzword Bingo
- Clustering
  - infers groups based on clustered objects
  - the process of grouping a set of objects (documents) into classes of similar objects (documents)
  - most common form of unspervised learning
  - Docs in same cluster behave similar with respect to relevance to information needs
  - Applications
    - Speed up vector space retrieval
    - imporved recall => better search results
  - Requirements
    - ![](img/req.PNG)
  - Problems
    - ![](img/prob.PNG)
  - Documents within a cluster should be similar
  - Documents from different clusters should be dissimilar
  - Algos
    - Distance Based
      - K Means
      - single-pass
    - Hirarchical
      - Bottom Up
      - Top Down
    - Other
      - Suffix Tree Clustering
- Pipeline
  - Partitioning Algo
    - create k clusters
- K-Means
  - clusters based on centroids
  - Reassignment of instances to clusters is based on distance to the current cluster centroids
- Hard Clustering
  - one doc => one cluster
  - K-means
- Soft Clustering
  - one doc => set of clusters
  - gives a probability that a doc belongs to a specific cluster
  - Sum of possibilties = 1
  - Types
    - Fuzzy Clustering (pattern recognition)
    - soft K-means
- Naive Bayes Model
  - ![](./img/bayes.PNG)
- Expectation Maximization Algo
  - uses bayes model
  - Expectation
    - Use naive bayes to compute probability => soft label
  - Maximization
    - use standard naives bayes training to learn to re-estimate params
- Hirarchical clustering - HAC
  - ![](img/den.PNG)
  - ![](img/hac.PNG)
- Buckshot Algo
  - HAC + K-means
- When is clustering good?
  - nodes in a cluster similar (intra-calss similarity = high)
  - nodes in other clusters different (inter-class similarity = low)
- Cluster Quality Evaluation
  - purity
    - ratio of dominant class and the size of the cluster
  - entropy of classes
    - mutual information
- Silhouette Values
  - Good clusters have the property that cluster members are close to each other and far from members of other clusters
- 

- Classification
  - assigns objects to predefined groups
  - Rocchio Method
    - tf-idf weights
    - assign to closest centroid
  - k Nearest Neighbor kNN
# Information Extraction

# Web Search & Crawling

# Semantic Knowledge Models, Semantic Web Stack

# Knowledge Engineering & Onthology Description

# OWL DL Example

# Queries und Ausdruck
