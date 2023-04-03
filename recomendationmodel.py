import pandas as pd
import gensim
import numpy as np
df = pd.read_csv(r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\test.csv')
print(df.shape)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
#  1st step for removing stop words is to use the library and remove stop words, this will make 
# sure that some basic stop words and numerical values is been removed from the Benefits column.

# 2nd step is that there might be stop words present in upper case for example "This" is a stop words 
# which was not removed in 1st step so I converted the Benefits column to lower case. Moreover we can only  
# lowercase any sentence or entire column when that column is free of any numerical value.

# 3rd then further removing stopwords, in this way we get the whole benefits column free from stop words

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
df['Benefits'] = df['Benefits'].apply(str.lower)
df['Benefits']=df.Benefits.apply(remove_stopwords)
df['Benefits']=df.Benefits.apply(strip_non_alphanum)
df['Benefits']=df.Benefits.apply(strip_numeric)
df['Benefits']=df.Benefits.apply(strip_multiple_whitespaces)
df['Asana']=df.Asana.apply(strip_multiple_whitespaces)
df['Asana'] = df['Asana'].apply(str.lower)
# print(df)
benefits = df['Benefits'].apply(gensim.utils.simple_preprocess)
# print(benefits)
print(benefits[0])
model = gensim.models.Word2Vec(
    window=5,
    min_count=2,
    workers=4
)
model.build_vocab(benefits, progress_per=5)
model.train(benefits, total_examples=model.corpus_count, epochs=2000)
model.wv.most_similar("sciatica", topn= 100)

print((model.wv.get_vector('pain')))
model.corpus_count
# model.save("model.bin")
from gensim.models import Word2Vec

# Load the Word2Vec model
model = Word2Vec.load(r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\model.bin')

# Get the list of unique words in the vocabulary
words = list(model.wv.key_to_index.keys())
# print(words)
print(len(words))
dict_of_word_embeddings = dict({})
for i in words:
    dict_of_word_embeddings[i] = model.wv[i]
# print(dict_of_word_embeddings)
Unique_words = dict_of_word_embeddings.keys()
word_vectors  = dict_of_word_embeddings.values()
# print(asanas, word_vectors)
d = {'Unique_words' : Unique_words , 'Word_Vectors' : word_vectors}
dataframe = pd.DataFrame(data = d)
dataframe
asanas = list(df['Asana'])
# total asanas present ( with repetition)
print(len(asanas))
asana = []
      

for x in asanas:
  if x not in asana:
    asana.append(x)
# total number of unique asanas
print(len(asana))    
# list of unique asanas
print(asana)


from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(asana)
#print(integer_encoded)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

### One hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# onehot_encoded

asan_dict={}

for i in range(len(asana)):
  asan_dict[asana[i]] = onehot_encoded[i]

# print(asan_dict) 
from tensorflow import keras
from tensorflow.keras.layers import Dense
print(benefits[0])
pair=[]

i=0
a=len(asana)
for x in benefits:
  if(i<a):
    target=asana[i]
    for y in x:
      if(y not in words):
        continue
      pair.append((y,target))
  i+=1  
print(pair)
contexts=[dict_of_word_embeddings[context] for context,target in pair]
contexts=np.vstack(contexts)
# shape of the context words matrix
contexts.shape
targets=[asan_dict[target] for context,target in pair]
targets=np.vstack(targets)
# shape of the target words matrix

targets.shape
from tensorflow import keras
from tensorflow.keras.layers import Dense

network_input = keras.Input(shape=contexts.shape[1], name='input_layer')
                                                                                    # Create a hidden layer for the network; store under 'hidden_layer'
hidden_layer1 = Dense(units=1000, activation='sigmoid', name='hidden_layer1')(network_input)

                                                                                            # Create an output layer for the network; store under 'output_layer'
output_layer = Dense(units=targets.shape[1], activation='softmax', name='output_layer')(hidden_layer1)

                                                                          # Create a Keras Model; store under 'embedding_model'
embedding_model = keras.Model(inputs=network_input, outputs=output_layer)

                                                          # Compile the model for training; define loss function
embedding_model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

                                                          # Print out a summary of the model
embedding_model.summary()
emb_model = embedding_model.fit(x=contexts,   # inputs
                    y=targets,   # outputs
                    batch_size=1024,  # how many pairs of words processed simultaneously
                    epochs=100,   # how many times we loop through the whole data
                    verbose=1   # do not print training status
                   )

import pickle
pickle.dump(emb_model,open('savemodel.sav','wb'))
from collections import Counter
from IPython.display import clear_output

def magic(user_input_words):
  predicted_asanas = []
#   user_input_words= []
  final_predicted_asanas = []
#   number_in_words = ['first','second', 'third', 'fourth']
#   for i in range(4):
#     user_input_words.append(input(f"Enter {number_in_words[i]} benefit word:  "))
  for i in user_input_words:
    if i in dict_of_word_embeddings:

      input_array = np.expand_dims(dict_of_word_embeddings[i], axis=0)
      prediction = embedding_model.predict(input_array)
      flatten_pred = prediction.flatten()
      result_indices = flatten_pred.argsort()[-10:][::-1]
    
      for result in result_indices:
        predicted_asanas.append(asana[result])
    
    
  counter_found = Counter(predicted_asanas)
  final_predicted_asanas_with_freq = counter_found.most_common(7)
  # print(final_predicted_asanas_with_freq)

  for yoga, freq in final_predicted_asanas_with_freq:
    final_predicted_asanas.append(yoga)
  
  return final_predicted_asanas