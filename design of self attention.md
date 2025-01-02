
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

```

