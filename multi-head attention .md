you may have experience that, your manager ask the whole team to join a meeting for brain storm for a single topic. Why we need many people to discuss problem together? Because we have limit of our own.We 
are confine to our own capability, knowledge base, personal experience, therefore when looking at a problem, we always have our own bias or blind spot, and we can't solve any problem by our own and 
view point of others would bring new idear, new view point and bring new hope for solving the problem.

The same principle apply to attention mechanism, if we only use only one attention, we may only capture some part of info from the sentence, and the algorithm may loss important info from the sentence. If
we simulate the human meeting, that is we compute multiple batch of attention weights at parallel and combine all those attention weights together, then we can capture more precise info for the given 
sentence or artice and prevent the limit or bias of one attention weights.

Computing a batch of attention weights and combine them together is called multi-head attention let's see how to use code to do it:

```py
'''
multi-head attention is doing marked attention for several times and combine those results together
'''
import torch.nn as nn
class MultiHeadAttention(nn.Module):
  def __init__(self, input_vec_length, output_vec_length, word_count_in_sentence, 
               num_heads ,bias = False):
    super().__init__()
    #concate several MarkedAttention in a list 
    marked_attn_list = []
    for _ in range(0, num_heads):
      marked_attn_list.append(
          MarkedAttention(input_vec_length, output_vec_length, word_count_in_sentence, bias)
      )
    self.heads = nn.ModuleList(marked_attn_list)

  def forward(self, input):
    attn_weight_list = []
    for head in self.heads:
      attn_weight_list.append(head(input))
    
    '''
    one attention weight is a vector of length 2,
    then if we have two heads, we will have two vectors each has length of 2,
    then we concate then into a vector with a length of 4
    '''
    return torch.cat(attn_weight_list, dim = -1)

```
Now let's try to run above code as following:
```py
#construct two sentences
batch = torch.stack([inputs, inputs], dim = 0)
print(f"batch: {batch}")
word_count_in_sentence = batch.shape[1] #the row of the matrix
word_vec_length = inputs.shape[1] #length of the vector of word
new_word_vec_length = 2

mha = MultiHeadAttention(word_vec_length, new_word_vec_length, word_count_in_sentence, 2, False)
new_word_vecs = mha(batch)
print(f"new word vecs:{new_word_vecs}")
```
The result of running above code is :
```py
batch: tensor([[[0.1000, 0.2000, 0.3000],
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
attn scores before marking: tensor([[[-0.0152, -0.0129, -0.0107, -0.0084, -0.0061, -0.0038, -0.0016,
           0.0007],
         [-0.0204, -0.0168, -0.0133, -0.0097, -0.0062, -0.0026,  0.0009,
           0.0045],
         [-0.0256, -0.0208, -0.0159, -0.0111, -0.0063, -0.0014,  0.0034,
           0.0082],
         [-0.0308, -0.0247, -0.0186, -0.0125, -0.0064, -0.0002,  0.0059,
           0.0120],
         [-0.0360, -0.0286, -0.0212, -0.0138, -0.0064,  0.0010,  0.0083,
           0.0157],
         [-0.0412, -0.0325, -0.0239, -0.0152, -0.0065,  0.0022,  0.0108,
           0.0195],
         [-0.0464, -0.0364, -0.0265, -0.0165, -0.0066,  0.0033,  0.0133,
           0.0232],
         [-0.0516, -0.0404, -0.0291, -0.0179, -0.0067,  0.0045,  0.0158,
           0.0270]],

        [[-0.0152, -0.0129, -0.0107, -0.0084, -0.0061, -0.0038, -0.0016,
           0.0007],
         [-0.0204, -0.0168, -0.0133, -0.0097, -0.0062, -0.0026,  0.0009,
           0.0045],
         [-0.0256, -0.0208, -0.0159, -0.0111, -0.0063, -0.0014,  0.0034,
           0.0082],
         [-0.0308, -0.0247, -0.0186, -0.0125, -0.0064, -0.0002,  0.0059,
           0.0120],
         [-0.0360, -0.0286, -0.0212, -0.0138, -0.0064,  0.0010,  0.0083,
           0.0157],
         [-0.0412, -0.0325, -0.0239, -0.0152, -0.0065,  0.0022,  0.0108,
           0.0195],
         [-0.0464, -0.0364, -0.0265, -0.0165, -0.0066,  0.0033,  0.0133,
           0.0232],
         [-0.0516, -0.0404, -0.0291, -0.0179, -0.0067,  0.0045,  0.0158,
           0.0270]]], grad_fn=<UnsafeViewBackward0>)
attn scores after marking: tensor([[[-0.0152,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0204, -0.0168,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0256, -0.0208, -0.0159,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0308, -0.0247, -0.0186, -0.0125,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0360, -0.0286, -0.0212, -0.0138, -0.0064,  0.0000,  0.0000,
           0.0000],
         [-0.0412, -0.0325, -0.0239, -0.0152, -0.0065,  0.0022,  0.0000,
           0.0000],
         [-0.0464, -0.0364, -0.0265, -0.0165, -0.0066,  0.0033,  0.0133,
           0.0000],
         [-0.0516, -0.0404, -0.0291, -0.0179, -0.0067,  0.0045,  0.0158,
           0.0270]],

        [[-0.0152,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0204, -0.0168,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0256, -0.0208, -0.0159,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0308, -0.0247, -0.0186, -0.0125,  0.0000,  0.0000,  0.0000,
           0.0000],
         [-0.0360, -0.0286, -0.0212, -0.0138, -0.0064,  0.0000,  0.0000,
           0.0000],
         [-0.0412, -0.0325, -0.0239, -0.0152, -0.0065,  0.0022,  0.0000,
           0.0000],
         [-0.0464, -0.0364, -0.0265, -0.0165, -0.0066,  0.0033,  0.0133,
           0.0000],
         [-0.0516, -0.0404, -0.0291, -0.0179, -0.0067,  0.0045,  0.0158,
           0.0270]]], grad_fn=<MaskedFillBackward0>)
new word vecs:tensor([[[-0.7797, -0.0746,  0.5116,  0.3055],
         [-0.7507, -0.0698,  0.3205,  0.1934],
         [-0.8693, -0.0787,  0.8590,  0.5157],
         [-0.7047, -0.0632,  0.6942,  0.4201],
         [-0.7716, -0.0619,  0.4610,  0.2737],
         [-0.2755, -0.0252,  0.6178,  0.3721],
         [-0.4806, -0.0478,  0.3824,  0.2244],
         [-0.6314, -0.0601,  0.7578,  0.4522]],

        [[-0.6223, -0.0483,  0.4671,  0.2815],
         [-0.5752, -0.0515,  0.7810,  0.4667],
         [-0.8924, -0.0710,  0.7112,  0.4269],
         [-0.4151, -0.0481,  0.6088,  0.3625],
         [-0.4091, -0.0243,  0.1140,  0.0637],
         [-0.4476, -0.0418,  0.2352,  0.1358],
         [-0.9206, -0.0742,  0.1645,  0.0954],
         [-0.6384, -0.0523,  0.6638,  0.3971]]], grad_fn=<CatBackward0>)
```
Compare with previous output, we have two three dimensional matries in the final list, and each matrix conresponding to output of one head.
