If you read those paper related to llm, you may come into a word like marked attention or causal attention. This mechanism is used to restrict the model to look for words before the next predict word. For example for sentence like
"jim likes to play football", if the task is going to predict the word "football", then the marked attention will ask the model only pay attention to words as "jim", "likes", "to", "play", it can not look into the word or beyond the word
it needs to  predict.

Comparing with previous section, we allow the current word to compute its attetion scores with thoses words appearing behide it. The process for marked attention is just like following:

to create new vector for word "jim", the sentence is:

jim x x x x,

to create new vector for word "likes", the sentence is:

jim likes x x x.

The process going on as above, the mark symbol "x" tells the model you can't take such symbol into consideration when you computing attention socres for current word. We can archive this base on our previous code, that is we are going to
make any attentions scores for words that are behide the current predicting word. Let me show you how to do it by using following code:

```py
import torch
import torch.nn as nn 

import torch.nn as nn


class SelfAttentionV2_marked_attetion(nn.Module):
  def __init__(self, input_vec_length, output_vec_length, bias = False):
    super().__init__()
    self.W_query = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_key = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_value = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)

  def forward(self, inputs):
    #for looping implicityly
    keys = self.W_key(inputs)
    values = self.W_value(inputs)
    query = self.W_query(inputs)
    attn_scores = query @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
    #how many words in the sentence
    vec_length = attn_scores.shape[0]
    #create diagonal matrix with up half to 0 and lower half to 1
    attn_mask = torch.tril(torch.ones(vec_length, vec_length))
    print(f"attn mask: {attn_mask}")
    '''
    the * operator for two matrix is not matrix multiplication like @, 
    given two vector [a,b,c], [e,f,g], the result of [a,b,c] * [e,f,g]
    is [a*e, b*f, c*g]
    '''
    masked_attn_weights = attn_weights * attn_mask
    print(f"maskt attn weights:{masked_attn_weights}")
    #normalize again, since the
    masked_attn_weights = torch.softmax(masked_attn_weights / keys.shape[-1] ** 0.5, dim = -1)
    new_word_vecs = masked_attn_weights @ values
    return new_word_vecs
```

Then we can run above code by following code:

```py
torch.manual_seed(123)
attn_process = SelfAttentionV2_marked_attetion(3, 2)
attn_process.forward(inputs)
```
And the given result is :

```py
attn mask: tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])
maskt attn weights:tensor([[0.1206, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1186, 0.1203, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1166, 0.1189, 0.1212, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1146, 0.1174, 0.1203, 0.1233, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1127, 0.1160, 0.1194, 0.1229, 0.1265, 0.0000, 0.0000, 0.0000],
        [0.1107, 0.1145, 0.1185, 0.1225, 0.1267, 0.1311, 0.0000, 0.0000],
        [0.1088, 0.1131, 0.1175, 0.1221, 0.1269, 0.1319, 0.1371, 0.0000],
        [0.1070, 0.1117, 0.1166, 0.1217, 0.1271, 0.1327, 0.1386, 0.1447]],
       grad_fn=<MulBackward0>)
tensor([[-0.0220,  0.0017],
        [-0.0563,  0.0012],
        [-0.1032, -0.0014],
        [-0.1635, -0.0062],
        [-0.2386, -0.0133],
        [-0.3303, -0.0230],
        [-0.4412, -0.0356],
        [-0.5748, -0.0515]], grad_fn=<MmBackward0>)
```
As we can see the attn mark is a diagonal matrix, and when we use the attn_weight matrix to do the operator "*", then we can ignore the attention scores coming from words after the current word.

As we can see above code that it is doing softmat twice, we can improve it by using only one softmax as following:
```py
import torch
import torch.nn as nn

import torch.nn as nn

#improve to do the noramlization at one time
class SelfAttentionV2_marked_attetion_improved(nn.Module):
  def __init__(self, input_vec_length, output_vec_length, bias = False):
    super().__init__()
    self.W_query = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_key = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_value = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)

  def forward(self, inputs):
    #for looping implicityly
    keys = self.W_key(inputs)
    values = self.W_value(inputs)
    query = self.W_query(inputs)
    attn_scores = query @ keys.T
    vec_length = attn_scores.shape[0]
    mark = torch.triu(torch.ones(vec_length, vec_length), diagonal=1)
    print(f"mask is :{mark}")
    marked_attn_scores = attn_scores.masked_fill(mark.bool(), -torch.inf)
    print(f"attn scores after marsked: {marked_attn_scores}")
    '''
    the trick here is: for softmax([a,b,c]) => sum = e^a+e^b+e^c, 
    [e^a/sum, e^b/sum, e^c/sum]
    if a, b or c is minus infinity, then e^a => 0 which is the same as marked
    '''
    attn_weights = torch.softmax(marked_attn_scores / keys.shape[-1] ** 0.5, dim = -1)
   
    new_word_vecs = attn_weights @ values
    return new_word_vecs
```
Something need to be noticed is, we change tril to triu, tril used to make lower part of the matrix with 1, and triu used to make the up part of matrix to 1, we can execute above code :

```py
torch.manual_seed(123)
attn_process = SelfAttentionV2_marked_attetion_improved(3, 2)
attn_process.forward(inputs)
```
above code given following output:
```py
mask is :tensor([[0., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
attn scores after marsked: tensor([[0.0281,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.0411, 0.0622,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.0541, 0.0818, 0.1096,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.0670, 0.1014, 0.1359, 0.1703,   -inf,   -inf,   -inf,   -inf],
        [0.0800, 0.1210, 0.1621, 0.2032, 0.2443,   -inf,   -inf,   -inf],
        [0.0929, 0.1406, 0.1884, 0.2361, 0.2838, 0.3315,   -inf,   -inf],
        [0.1059, 0.1603, 0.2146, 0.2690, 0.3234, 0.3777, 0.4321,   -inf],
        [0.1188, 0.1799, 0.2409, 0.3019, 0.3629, 0.4239, 0.4850, 0.5460]],
       grad_fn=<MaskedFillBackward0>)
tensor([[-0.1827,  0.0140],
        [-0.2357,  0.0051],
        [-0.2893, -0.0038],
        [-0.3437, -0.0129],
        [-0.3992, -0.0222],
        [-0.4561, -0.0317],
        [-0.5145, -0.0415],
        [-0.5748, -0.0515]], grad_fn=<MmBackward0>)
```
Looking at above output, we can see the mask is just opposite of previous before, the 1 in this mark is 0 at previous mark, and the code use attn_scores.masked_fill(mark.bool(), -torch.inf) to make every elements of attn_scores that are
in the position of 1 into minus infinity. Then when doing softmax, we will put the value as the power of e, and when the minus infinity is the power of e, the result will turn into 0 which is the same as using 0 to remove the inference
of given attention score.

In order to improve the training process, we need to apply some trick called "dropout", it just randomly set some vaules in the model to 0, for example given vector [a, b, c, d, e, f, g], if we set the level of drop out to 0.5, then
we will simple select half of all elmenets from the vector and set them to 0 and set those remaining with scale factor of 1/(1-p), p = 0.5, then the scaling factor is 2 which means the vector will turns into [2a, 0, 2c, 2d, 0,0,2g],
let's see an example of dropout with code as following

```py
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #set half elements to 0 randomly
example = torch.ones(6,6)
print(f"before dropout: {example}")
print(f"after dropot: {dropout(example)}")
```
As we can see the dropout factor is 0.5, then we will randomly set half of elements in the example matrix to 0, and scale up the remaining elements by factor of 1/(1-p)=2, then we get following result:

```py
before dropout: tensor([[1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.]])
after dropot: tensor([[2., 2., 2., 2., 2., 2.],
        [0., 2., 0., 0., 0., 0.],
        [0., 0., 2., 0., 2., 0.],
        [2., 2., 0., 0., 0., 2.],
        [2., 0., 0., 0., 0., 2.],
        [0., 2., 0., 0., 0., 0.]])
```
Now let's apply the dropout to the marked attention weights as following:

```py
import torch
import torch.nn as nn

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #set half elements to 0 randomly

class SelfAttentionV2_marked_attetion_improved_dropout(nn.Module):
  def __init__(self, input_vec_length, output_vec_length, bias = False):
    super().__init__()
    self.W_query = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_key = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_value = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)

  def forward(self, inputs):
    #for looping implicityly
    keys = self.W_key(inputs)
    values = self.W_value(inputs)
    query = self.W_query(inputs)
    attn_scores = query @ keys.T
    vec_length = attn_scores.shape[0]
    mark = torch.triu(torch.ones(vec_length, vec_length), diagonal=1)
    print(f"mask is :{mark}")
    marked_attn_scores = attn_scores.masked_fill(mark.bool(), -torch.inf)
    print(f"attn scores after marsked: {marked_attn_scores}")
    '''
    the trick here is: for softmax([a,b,c]) => sum = e^a+e^b+e^c, 
    [e^a/sum, e^b/sum, e^c/sum]
    if a, b or c is minus infinity, then e^a => 0 which is the same as marked
    '''
    attn_weights = torch.softmax(marked_attn_scores / keys.shape[-1] ** 0.5, dim = -1)
    print(f"attn weights before dropout:{attn_weights}")
    dropout_attn_weights = dropout(attn_weights)
    print(f"attn weights after drop out:{dropout_attn_weights}")
   
    new_word_vecs = dropout_attn_weights @ values
    return new_word_vecs

attn_process = SelfAttentionV2_marked_attetion_improved_dropout(3, 2)
attn_process.forward(inputs)
```
Running above code we get the following :
```py
marked attn weights with dropout: tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2371, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2332, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2388, 0.2459, 0.2531, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.2291, 0.0000, 0.0000, 0.0000, 0.2622, 0.0000, 0.0000],
        [0.2177, 0.0000, 0.0000, 0.0000, 0.2539, 0.2638, 0.0000, 0.0000],
        [0.2139, 0.2233, 0.0000, 0.0000, 0.0000, 0.2654, 0.0000, 0.2893]],
       grad_fn=<MulBackward0>)
tensor([[ 0.0000,  0.0000],
        [-0.0433,  0.0033],
        [-0.0426,  0.0033],
        [ 0.0000,  0.0000],
        [-0.3692, -0.0289],
        [-0.2518, -0.0202],
        [-0.3800, -0.0308],
        [-0.5575, -0.0490]], grad_fn=<MmBackward0>)
```

Until now, our classes above can only receive one vector, that is the inputs can only be one matrix and its each row conresponding to each word vector, this is low effieciency, we need to enable to handle multiple sentences at the sametime,
Since one sentence is one 2 dimensional matrix, multiple sentences means we need to handle a vector witl many elements and each element is a matrix. This structure will turn into a three dimensional matrix and it is more difficult to understand
compare with 2 dimension matrix.

Let me give you some instinc feeling for three dimensional matrix by code as following:

```py
#example of three dimensional matrix
batch = torch.stack((inputs, inputs))
print(f"three dimensional matrix: {batch}, \nwith shape as {batch.shape}")
```
Running the code you will get following result:
```py
three dimensional matrix: tensor([[[0.1000, 0.2000, 0.3000],
         [0.2000, 0.3000, 0.4000],
         [0.3000, 0.4000, 0.5000],
         [0.4000, 0.5000, 0.6000],
         [0.5000, 0.6000, 0.7000],
         [0.6000, 0.7000, 0.8000],
         [0.7000, 0.8000, 0.9000],
         [0.8000, 0.9000, 1.0000]],

        [[0.1000, 0.2000, 0.3000],
         [0.2000, 0.3000, 0.4000],
         [0.3000, 0.4000, 0.5000],
         [0.4000, 0.5000, 0.6000],
         [0.5000, 0.6000, 0.7000],
         [0.6000, 0.7000, 0.8000],
         [0.7000, 0.8000, 0.9000],
         [0.8000, 0.9000, 1.0000]]])
with shape as torch.Size([2, 8, 3])
```
As we can see, that is a list with 2 elements, and each element of the list is a two dimensional matrix with row number of 8 and column number of 3. Now let's wrap all things together as following:

```py
import torch
import torch.nn as nn

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #set half elements to 0 randomly

class MarkedAttention(nn.Module):
  def __init__(self, input_vec_length, output_vec_length, word_count_in_sentence ,bias = False):
    super().__init__()
    self.W_query = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_key = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)
    self.W_value = nn.Linear(in_features = input_vec_length, out_features = output_vec_length, bias = bias)

    '''
    This is related to how torch handle memory with multiple gpu, it is optimization
    for engineering purpose
    '''
    self.register_buffer("mask", torch.triu(torch.ones(word_count_in_sentence, 
                                                       word_count_in_sentence), diagonal=1))
    print(f"register buffer:{self.register_buffer}")

  def forward(self, inputs):
    '''
    inputs here is dimensional 3,
    '''
    b, num_tokens, d_in = inputs.shape
    keys = self.W_key(inputs)
    values = self.W_value(inputs)
    query = self.W_query(inputs)
    #we can't use keys.T anymore because keys is three dimensional
    attn_scores = query @ keys.transpose(1,2)

    #doing marking here, marking before softmax or after softmax make no different
    print(f"attn scores before marking: {attn_scores}")
    '''
    the difference of masked_fill_ which has an underscore and masked_fill is:
    masked_fill_ will change the given tensor directly, but masked_fill will
    make a new copy of the given tensor and change the copied tensor that is 
    more slower than masked_fill_
    '''
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], 0)
    print(f"attn scores after marking: {attn_scores}")

    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
    #how many words in the sentence
    vec_length = attn_scores.shape[0]


    masked_attn_weights_dropout = dropout(attn_weights)
    #apply dropout on marked attention weights
    print(f"marked attn weights with dropout: {masked_attn_weights_dropout}")

    new_word_vecs = masked_attn_weights_dropout @ values
    return new_word_vecs

```

Then we will execute the above code as following:
```py
batch = torch.stack((inputs, inputs), dim=0)
word_count_in_sentence = batch.shape[1]
word_vec_length = inputs.shape[1]
new_word_vec_length = 2

ma = MarkedAttention(word_vec_length, new_word_vec_length, word_count_in_sentence, 0.0)
ma = MarkedAttention(word_vec_length, new_word_vec_length, word_count_in_sentence, 0.0)
new_word_vecs = ma(batch)

print(f"new word vecs:{new_word_vecs}")
```
Running above code we get following result:

```py
register buffer:<bound method Module.register_buffer of MarkedAttention(
  (W_query): Linear(in_features=3, out_features=2, bias=False)
  (W_key): Linear(in_features=3, out_features=2, bias=False)
  (W_value): Linear(in_features=3, out_features=2, bias=False)
)>
attn scores before marking: tensor([[[0.0281, 0.0426, 0.0571, 0.0716, 0.0861, 0.1005, 0.1150, 0.1295],
         [0.0411, 0.0622, 0.0834, 0.1045, 0.1256, 0.1467, 0.1679, 0.1890],
         [0.0541, 0.0818, 0.1096, 0.1374, 0.1652, 0.1929, 0.2207, 0.2485],
         [0.0670, 0.1014, 0.1359, 0.1703, 0.2047, 0.2391, 0.2736, 0.3080],
         [0.0800, 0.1210, 0.1621, 0.2032, 0.2443, 0.2853, 0.3264, 0.3675],
         [0.0929, 0.1406, 0.1884, 0.2361, 0.2838, 0.3315, 0.3793, 0.4270],
         [0.1059, 0.1603, 0.2146, 0.2690, 0.3234, 0.3777, 0.4321, 0.4865],
         [0.1188, 0.1799, 0.2409, 0.3019, 0.3629, 0.4239, 0.4850, 0.5460]],

        [[0.0281, 0.0426, 0.0571, 0.0716, 0.0861, 0.1005, 0.1150, 0.1295],
         [0.0411, 0.0622, 0.0834, 0.1045, 0.1256, 0.1467, 0.1679, 0.1890],
         [0.0541, 0.0818, 0.1096, 0.1374, 0.1652, 0.1929, 0.2207, 0.2485],
         [0.0670, 0.1014, 0.1359, 0.1703, 0.2047, 0.2391, 0.2736, 0.3080],
         [0.0800, 0.1210, 0.1621, 0.2032, 0.2443, 0.2853, 0.3264, 0.3675],
         [0.0929, 0.1406, 0.1884, 0.2361, 0.2838, 0.3315, 0.3793, 0.4270],
         [0.1059, 0.1603, 0.2146, 0.2690, 0.3234, 0.3777, 0.4321, 0.4865],
         [0.1188, 0.1799, 0.2409, 0.3019, 0.3629, 0.4239, 0.4850, 0.5460]]],
       grad_fn=<UnsafeViewBackward0>)
attn scores after marking: tensor([[[0.0281, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0411, 0.0622, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0541, 0.0818, 0.1096, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0670, 0.1014, 0.1359, 0.1703, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0800, 0.1210, 0.1621, 0.2032, 0.2443, 0.0000, 0.0000, 0.0000],
         [0.0929, 0.1406, 0.1884, 0.2361, 0.2838, 0.3315, 0.0000, 0.0000],
         [0.1059, 0.1603, 0.2146, 0.2690, 0.3234, 0.3777, 0.4321, 0.0000],
         [0.1188, 0.1799, 0.2409, 0.3019, 0.3629, 0.4239, 0.4850, 0.5460]],

        [[0.0281, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0411, 0.0622, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0541, 0.0818, 0.1096, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0670, 0.1014, 0.1359, 0.1703, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0800, 0.1210, 0.1621, 0.2032, 0.2443, 0.0000, 0.0000, 0.0000],
         [0.0929, 0.1406, 0.1884, 0.2361, 0.2838, 0.3315, 0.0000, 0.0000],
         [0.1059, 0.1603, 0.2146, 0.2690, 0.3234, 0.3777, 0.4321, 0.0000],
         [0.1188, 0.1799, 0.2409, 0.3019, 0.3629, 0.4239, 0.4850, 0.5460]]],
       grad_fn=<MaskedFillBackward0>)
marked attn weights with dropout: tensor([[[0.0000, 0.0000, 0.2494, 0.2494, 0.0000, 0.2494, 0.2494, 0.0000],
         [0.2550, 0.0000, 0.2477, 0.0000, 0.2477, 0.2477, 0.0000, 0.0000],
         [0.2541, 0.0000, 0.0000, 0.2445, 0.2445, 0.2445, 0.2445, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2395, 0.2395, 0.2395],
         [0.0000, 0.0000, 0.2604, 0.2681, 0.2760, 0.2322, 0.0000, 0.2322],
         [0.0000, 0.2459, 0.0000, 0.0000, 0.0000, 0.2815, 0.0000, 0.0000],
         [0.2271, 0.0000, 0.0000, 0.0000, 0.2648, 0.2752, 0.0000, 0.2107],
         [0.2139, 0.2233, 0.0000, 0.0000, 0.0000, 0.2654, 0.0000, 0.2893]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.2494, 0.2494, 0.2494, 0.0000],
         [0.0000, 0.0000, 0.2477, 0.2477, 0.2477, 0.2477, 0.0000, 0.2477],
         [0.0000, 0.0000, 0.2642, 0.0000, 0.2445, 0.2445, 0.2445, 0.0000],
         [0.0000, 0.2573, 0.0000, 0.0000, 0.2395, 0.2395, 0.0000, 0.0000],
         [0.2457, 0.0000, 0.2604, 0.2681, 0.0000, 0.0000, 0.2322, 0.0000],
         [0.0000, 0.2459, 0.0000, 0.0000, 0.0000, 0.0000, 0.2226, 0.0000],
         [0.2271, 0.2360, 0.0000, 0.2549, 0.0000, 0.2752, 0.0000, 0.0000],
         [0.2139, 0.2233, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],
       grad_fn=<MulBackward0>)

new word vecs:tensor([[[-0.6021, -0.0562],
         [-0.4690, -0.0340],
         [-0.6882, -0.0602],
         [-0.5848, -0.0658],
         [-0.7807, -0.0740],
         [-0.2703, -0.0217],
         [-0.5901, -0.0551],
         [-0.5575, -0.0490]],

        [[-0.5303, -0.0553],
         [-0.7736, -0.0742],
         [-0.6238, -0.0598],
         [-0.3883, -0.0321],
         [-0.4699, -0.0337],
         [-0.2520, -0.0213],
         [-0.4315, -0.0279],
         [-0.1034,  0.0022]]], grad_fn=<UnsafeViewBackward0>)
```
