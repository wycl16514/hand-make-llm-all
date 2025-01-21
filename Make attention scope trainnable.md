In previous section, we show how to compute the attention scopes. Given a word from the sentence, we using its vector to do dot product with vector of other words in the sentece, then we normalize 
those dot product results, and finally we compute a new word vector for the selected word by adding vectors of words in the sentence with conresponding dot product value which is the attention scope.

The point is all the value are fixed, which means it can't make any progress even we have materials to train the model. In deep learning there is a paradim if you want to make some "dead" value 
"live" again, that is making any fixed value to be changenable, the way is to multiply a matrix with it. At the same time we make the value of the matrix changenable, then we can change the "fixed"
value by changing the matrix, let's see how we can use this methods to make the fixed attention scope changenable.

First we initialize three matries, W_q, W_k, W_v, for the given selected word which is the query in last section, we multiple its word vector with W_q, then we get a new vecotr for the selected word.
We need to make sure the dimension of W_q, W_k, W_v, since we will use this matries to multiply with word vector from the right, then the number of rows of these matries need to be the length of 
word vector, then we can decide the colunms for those matries, the bigger for the column number, the better for the finall result but you need to pay more cost to train the model.

In previous section, we make our word vector with length of 3, then in order to simplify the process, we can make the column number of the matries to 2(only for learning purpose), then the dimension
for W_q, W_k, W_v is (3, 2), let's initialize them by using following code :

```py
#initialize trainable matries of W_q, W_k, W_v
d_in = 3
d_out = 2
torch.manual_seed(123)
#requires_grad = False tell torch we will not going to train the matrix for now
W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(f"W_q: {W_q}\n W_k: {W_k}, W_v: {W_v}")
```

Run the above code we get following outputs:

```py
W_q: Parameter containing:
tensor([[0.2961, 0.5166],
        [0.2517, 0.6886],
        [0.0740, 0.8665]])
 W_k: Parameter containing:
tensor([[0.1366, 0.1025],
        [0.1841, 0.7264],
        [0.3153, 0.6871]]), W_v: Parameter containing:
tensor([[0.0756, 0.1966],
        [0.3164, 0.4017],
        [0.1186, 0.8274]])
```
Now let's compute the new word vector for the word of "jim" as following:

```py
#new word vector for jim by V_jim @ W_q
V_jim = inputs[0]
V_jim_new = V_jim @ W_q
print(f"new word vector for word 'jim' is :{V_jim_new}")
```
Running above code we get following result:

```py
new word vector for word 'jim' is :tensor([0.1021, 0.4493])
```

Now we use vector of words in the sentences to multiply the matrix of W_v and W_v, the we call those result vectors as keys and  values:

```py
#multiply vectors of words from the sentence with W_k, W_v
keys = inputs @ W_k
values = inputs @ W_v
print(f"shape of keys: {keys.shape}")
print(f"shape of values: {values.shape}")
```
Running above code we get following result:

```py
shape of keys: torch.Size([8, 2])
shape of values: torch.Size([8, 2])
```
Since we have 8 different words in the sentence, and each word multiple W_k and W_v result in vector of length 2, therefore the shape of keys and values have shape (8, 2). Now we can use the new
vector that is V_jim_new and keys above to compute a attention score just like previous section:

```py
#keys.T has shape (2, 8), the following multiply get (1, 8) which is a vector with length of 8,
#each value in the result is attention scope for word jim
attn_scores_jim = V_jim_new @ keys.T
print(f"attetion scopres for word jim: {attn_scores_jim}")
```
Running above code will get following result:

```py
attetion scopres for word jim: tensor([0.1773, 0.2519, 0.3265, 0.4012, 0.4758, 0.5504, 0.6250, 0.6996])
```

Now we need to normalize the value above, make their sum to 1 as following:

```py
'''
why need to divide each value with square root of the dimension value of W_k? This is some kind of art, by doing this can make the trainning process more 
efficient, there are lots of "non-AI" in the process of designing AI, the purpose of it is to avoid small gradients which will greatly slow down the progress of traning 
'''

dimension_w_k = keys.shape[-1] #2
attn_weights_jim = torch.softmax(attn_scores_jim / dimension_w_k ** 0.5, dim = -1)
print(f"attention scores of word 'jim' after normalization: {attn_weights_jim}")
```
Then we can compute the new word vecotr for word "jim" by using vectors from the values to multiply with the attention scores above and sum them together:

```py
#new word vector of jim
word_vec_jim = attn_weights_jim @ values
print(f"word vector for word 'jim': {word_vec_jim}")
```
Running above code we get the following result:

```py
word vector for word 'jim': tensor([0.2992, 0.8866])
```

We can apply the process for all the words in the sentence by wrapping the whole process into a class as following:
```py
'''
Three matries are bollowed from databases, the query used to indicate what you are looking for,
key used to confine the info in given scope, and value is the details in the given scope. For example
you are going to find some thing to watch, then the query can be "movie", the selections for 
key can be "action, hollow, love, documentary", if the key is "action", then the value can be 
list of names of action movies
'''
import torch.nn as nn
class SelfAttentionV1(nn.Module):
  def __init__(self, input_vec_length, output_vec_lenth):
    super().__init__()
    #randomize the value for three matries
    self.W_query = nn.Parameter(torch.rand(input_vec_length, output_vec_lenth))
    self.W_key = nn.Parameter(torch.rand(input_vec_length, output_vec_lenth))
    self.W_value = nn.Parameter(torch.rand(input_vec_length, output_vec_lenth))
  
  def forward(self, inputs):
    '''
    inputs are words for the sentences, each word in the sentence will go through the process
    above, then each word can be the query word
    '''
    keys = inputs @ self.W_key
    values = inputs @ self.W_value
    query = inputs @ self.W_query
    attn_scores = query @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
    new_word_vecs = attn_weights @ values
    return new_word_vecs
```
Then we can run above call by using following code:

```py
torch.manual_seed(123)
attn_process = SelfAttentionV1(3, 2)
attn_process.forward(inputs)
```
using above code we get following result:

```py
tensor([[0.2992, 0.8866],
        [0.3058, 0.9051],
        [0.3124, 0.9233],
        [0.3188, 0.9412],
        [0.3251, 0.9588],
        [0.3312, 0.9760],
        [0.3372, 0.9927],
        [0.3430, 1.0089]], grad_fn=<MmBackward0>)
```

