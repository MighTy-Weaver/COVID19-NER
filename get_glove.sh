if [ ! -d "/glove" ]; then
  mkdir /myfolder
fi
cd glove
wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
wget http://downloads.cs.stanford.edu/nlp/data/glove.42B.300d.zip
wget http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip

unzip *.zip