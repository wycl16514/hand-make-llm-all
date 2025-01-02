
There are tons of materials talking about mechanism of self attention, I read lots of them but I don't think they can speak this topic in clearer way. The point is, this is a complex topic, 
we need to handle it by a way of divide and conqure.

Let's look at an example sentence: "jim and john are brothers, jim likes football and john likes swimming". Now let me ask you following questions:

jim -> (),  john -> (), brothers -> (), (),

What could you think about which word can put into the paratheses? It's not difficult to figer out that:

jim -> (football), john -> (swimming), brothers -> (jim), (john)

The word at the left of -> we call it "query", and the words at the right of -> has name "value". As we can see in a sentence, when we pick up a word, it is very likely that there is another
word that can "match" to it, which means they have attention to each other. By our example when you read the sentence and then give you the word "jim" you will pay your attention to the other
word that is "football".

That means a sentence may construct some kind of relationship for words, as long as you konw the relationship, you will capture the meaning of the sentence. Therefore we say for a given word
in a sentence, it may bring strong attention to another word, and we need some mathmatical way to describe such "attention". The most simple way to compute the "attention" is using a number,
the higher value of the number, the strong attention between the given two words.

How to compute the value of "attention" between two words? As we have seen before, we convert each word to vector, then we can compute attention value of two words by computing the inner 
product of two vectors. For example we give each word in the sentence of following vectors:

jim -> [0.1, 0.2, 0.3], and -> [0.2, 0.3,0.4], john -> [0.3, 0.4, 0.5], are -> [0.4,0.5, 0.6], brothers -> [0.5, 0.6, 0.7], likes -> [0.6,0.7,0.8], football -> [0.7, 0.8, 0.9],
swimming -> [0.8, 0.9, 1.0]

then the attention value for jim and football is [0.1, 0.2, 0.3] . [0.7, 0.8, 0.9] = 0.1*0.7 + 0.2 * 0.8 + 0.3 * 0.9 . And we can see attention value of two words is not related to their
position, that is the attention value of (jim, football) and (football, jim) are the same.

Base on attetion value, we can change the value of word vector, for example we that jim out from the sentence, and compute its attetion value with each word of the sentence, and we can
compute the attention value of "jim" and other word as following:

a(jim, jim) , a(jim, and), a(jim, john), a(jim, are), a(jim, brothers), a(jim, likes), a(jim, football), a(jim, swimming),

a(jim, and) means the attention value of word "jim" and "and", and we use V(jim) to represent the vector of word "jim", then base on the attention value we can readjust the vector for
jim as:

V'(jim) = V(jim)* a(jim, jim) + V(and) * a(jim, and) + .... + V(swimming) * a(jim, swimming)

Now let's see how to use code to implement above scheme, first we initialize the vector for each word as following:

```py
#init vectors for each words:
import torch

inputs = torch.tensor(
    [
        [0.1, 0.2, 0.3], # jim
        [0.2, 0.3,0.4], # and 
        [0.3, 0.4, 0.5], #john 
        [0.4,0.5, 0.6], # are 
        [0.5, 0.6, 0.7], # brothers
        [0.6,0.7,0.8], # likes
        [0.7, 0.8, 0.9], #football 
        [0.8, 0.9, 1.0], #swimming
    ]
)
```

Then taking the vector for "jim", we can compute its attetion value with other words as following:

```py
query = inputs[0] #vector for "jim"
attn_scores_jim = torch.empty(inputs.shape[0]) 
for i, x_i in enumerate(inputs):
  attn_scores_jim[i] = torch.dot(query, x_i)
  
print(attn_scores_jim)
```
Running above code we get the following result:

```py
tensor([0.1400, 0.2000, 0.2600, 0.3200, 0.3800, 0.4400, 0.5000, 0.5600])
```
value 0.1400 is attention value of a(jim, jim), and 0.2000 is attention value of a(jim, and) and so on. Now we need to normalize the attention value that is make them sum to 1.0 by using following
code:

```py
#normalize attention value, make them sum up to 1.0
attn_weights_jim = attn_scores_jim / attn_scores_jim.sum()
print(attn_weights_jim)
```
Running the code above we get following:

```py
tensor([0.0500, 0.0714, 0.0929, 0.1143, 0.1357, 0.1571, 0.1786, 0.2000])
```
Normally we use softmax to do normalize, that is for x1, x2...,xn, softmax(x1) = [exp(x1) / (exp(x1) +exp(x2) + ... + exp(xn))], let's put it into code:

```py
#softmax for normalization
def softmax(x):
  return (torch.exp(x) / torch.exp(x).sum()).numpy()

attn_weights_jim_softmax = softmax(attn_scores_jim)
print(attn_weights_jim_softmax)
```
Running above code we get following result:

```py
[0.10037188 0.10657853 0.11316898 0.12016696 0.12759766 0.13548787
 0.14386596 0.15276214]
```
Of course torch already provides us the softmax utility and we can use it directly instead of doing our own:

```py
'''
dim = -1 means doing the softmax compution by the inner most dimension, since inputs is two dimension (row, column)
the inner most dimension is column, therefore we are doing softmax on each row
'''
attn_weights_jim_softmax = torch.softmax(attn_scores_jim, dim = -1)
print(attn_weights_jim_softmax)
print(f"sum is :{attn_weights_jim_softmax.sum()}")
```

Running aboved code we get following result:

```py
tensor([0.1004, 0.1066, 0.1132, 0.1202, 0.1276, 0.1355, 0.1439, 0.1528])
sum is :1.0
```

Now we can compute the adjusted vector of "jim" after considering the attention values:

```py
query = inputs[0] #jim
attn_jim_vector = torch.zeros(inputs.shape[-1])
for i, x_i in enumerate(inputs):
  #sum up each vector with attention values
  attn_jim_vector += attn_weights_jim_softmax[i] * x_i

print(attn_jim_vector)
```

Running above code we get following result:

```py
tensor([0.4814, 0.5814, 0.6814])
```

Now we can compute adjusted vector for each word by computing attention value by the given word and other words in the sentence and get the adjusted vector as we seen above:

```py
#we have 8 words in the sentence, each word can compute attention value with other words which result in vector length of 8
#since we have 8 words, then we have 8 vectors which has length of 8
attn_scores = torch.empty(8, 8)
for i, x_i in enumerate(inputs):
  for j, x_j in enumerate(inputs):
    attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)
```

Running above code we get following result:

```py
tensor([[0.1400, 0.2000, 0.2600, 0.3200, 0.3800, 0.4400, 0.5000, 0.5600],
        [0.2000, 0.2900, 0.3800, 0.4700, 0.5600, 0.6500, 0.7400, 0.8300],
        [0.2600, 0.3800, 0.5000, 0.6200, 0.7400, 0.8600, 0.9800, 1.1000],
        [0.3200, 0.4700, 0.6200, 0.7700, 0.9200, 1.0700, 1.2200, 1.3700],
        [0.3800, 0.5600, 0.7400, 0.9200, 1.1000, 1.2800, 1.4600, 1.6400],
        [0.4400, 0.6500, 0.8600, 1.0700, 1.2800, 1.4900, 1.7000, 1.9100],
        [0.5000, 0.7400, 0.9800, 1.2200, 1.4600, 1.7000, 1.9400, 2.1800],
        [0.5600, 0.8300, 1.1000, 1.3700, 1.6400, 1.9100, 2.1800, 2.4500]])
```
Then we do softmax for each row in above matrix as following:

```py
attn_weights = torch.softmax(attn_scores, dim = -1)
print(attn_weights)
```
Running above code we get following result:

```py
tensor([[0.1004, 0.1066, 0.1132, 0.1202, 0.1276, 0.1355, 0.1439, 0.1528],
        [0.0893, 0.0977, 0.1069, 0.1170, 0.1280, 0.1401, 0.1533, 0.1677],
        [0.0791, 0.0892, 0.1006, 0.1134, 0.1278, 0.1441, 0.1625, 0.1832],
        [0.0698, 0.0810, 0.0942, 0.1094, 0.1271, 0.1477, 0.1716, 0.1993],
        [0.0612, 0.0733, 0.0878, 0.1051, 0.1258, 0.1506, 0.1803, 0.2159],
        [0.0535, 0.0660, 0.0815, 0.1005, 0.1240, 0.1530, 0.1887, 0.2328],
        [0.0466, 0.0592, 0.0753, 0.0957, 0.1217, 0.1547, 0.1967, 0.2500],
        [0.0404, 0.0529, 0.0693, 0.0908, 0.1190, 0.1559, 0.2042, 0.2675]])
```
Then we can compute adjusted vector for each word in one matrix operation:

```py
all_adjusted_vecs = attn_weights @ inputs
print(all_adjusted_vecs)
```
Running above code we get following result:

```py
tensor([[0.4814, 0.5814, 0.6814],
        [0.4968, 0.5968, 0.6968],
        [0.5120, 0.6120, 0.7120],
        [0.5269, 0.6269, 0.7269],
        [0.5413, 0.6413, 0.7413],
        [0.5553, 0.6553, 0.7553],
        [0.5688, 0.6688, 0.7688],
        [0.5817, 0.6817, 0.7817]])
```

That's all about the first step to understand self-attention, the point is the attetion value for two words is fixed, in next section we will see how to make the attention value trainable.
