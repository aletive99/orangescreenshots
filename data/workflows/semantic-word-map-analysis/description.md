This workflow imports text documents from folders through the Import Documents. The obtained corpus is preprocessed with selected methods with the Preprocess Text. The obtained new Corpus is then visualized through a Word Cloud where the users can select words to be analyzed. The selected words are then transformed into a Corpus through its relative widget which passes it to the Document Embedding which embeds the input corpus into a vector space and appends the features to the corpus. The obtained corpus is then visualized in a two dimensional space through a t-SNE. The Distances between the data points are also calculated and used to perform Hierarchical Clustering. This workflow allows the user to select frequent words from an imported document and visualize and identify potential clusters of semantically related words.