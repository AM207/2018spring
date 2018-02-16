---
title: Iteration
shorttitle: Iteration
notebook: Iteration.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}

## Sequences and their Abstractions

#### What is a sequence?

Consider the notion of **Abstract Data Types**. 

The idea there is that one data type might be implemented in terms of another, or some underlying code, not even in python. 

As long as the interface and contract presented to the user is solid, we can change the implementation below. 

More on this later..

The **dunder methods** in python are used towards this purpose. 

In python a sequence is something that follows the sequence protocol. An example of this is a python list. 

This entails defining the `__len__` and `__getitem__` methods. 



```python
alist=[1,2,3,4]
len(alist)#calls alist.__len__
```





    4





```python
alist[2]#calls alist.__getitem__(2)
```





    3



Lists also support slicing. How does this work?



```python
alist[2:4]
```





    [3, 4]



To see this lets create a dummy sequence which shows us what happens. This sequence does not create any storage, it just implements the protocol



```python
class DummySeq:
    
    def __len__(self):
        return 42
    
    def __getitem__(self, index):
        return index
```




```python
d = DummySeq()
len(d)
```





    42





```python
d[5]
```





    5





```python
d[67:98]
```





    slice(67, 98, None)



Slicing creates a `slice object` for us of the form `slice(start, stop, step)` and then python calls `seq.__getitem__(slice(start, stop, step))`.

What about two dimensional indexing, if we wanted to create a two dimensional structure?



```python
d[67:98:2,1]
```





    (slice(67, 98, 2), 1)





```python
d[67:98:2,1:10]
```





    (slice(67, 98, 2), slice(1, 10, None))



As sequence writers, our job is to interpret these in `__getitem__`



```python
#taken from Fluent Python
import numbers, reprlib

class NotSoDummySeq:    
    def __init__(self, iterator):
        self._storage=list(iterator)
        
    def __repr__(self):
        components = reprlib.repr(self._storage)
        components = components[components.find('['):]
        return 'NotSoDummySeq({})'.format(components)
    
    def __len__(self):
        return len(self._storage)
    
    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._storage[index])
        elif isinstance(index, numbers.Integral): 
            return self._storage[index]
        else:
            msg = '{cls.__name__} indices must be integers' 
            raise TypeError(msg.format(cls=cls))

```




```python
d2 = NotSoDummySeq(range(10))
len(d2)
```





    10





```python
d2
```





    NotSoDummySeq([0, 1, 2, 3, 4, 5, ...])





```python
d[4]
```





    4





```python
d2[2:4]
```





    NotSoDummySeq([2, 3])





```python
d2[1,4]
```



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-15-ae2b261447b5> in <module>()
    ----> 1 d2[1,4]
    

    <ipython-input-10-bae9aa90bd14> in __getitem__(self, index)
         22         else:
         23             msg = '{cls.__name__} indices must be integers'
    ---> 24             raise TypeError(msg.format(cls=cls))
    

    TypeError: NotSoDummySeq indices must be integers


## From positions in an array to Iterators

One can simply follow the `next` pointers to the next **POSITION** in a linked list. This suggests an abstraction of the **position** or pointer to an **iterator**, an abstraction which allows us to treat arrays and linked lists with an identical interface. 

The salient points of this abstraction are:

- the notion of a `next` abstracting away the actual gymnastics of where to go next in a storage system
- the notion of a `first` to a `last` that `next` takes us on a journey from and to respectively

- we already implemented the sequence protocol, but 
- now we suggest an additional abstraction that is more fundamental than the notion of a sequence: the **iterable**.

### Iterators and Iterables in python

Just as a sequence is something implementing `__getitem__` and `__len__`, an **Iterable** is something implementing `__iter__`. 

`__len__` is not needed and indeed may not make sense. 

The following example is taken from Fluent Python



```python
import reprlib
class Sentence:
    def __init__(self, text): 
        self.text = text
        self.words = text.split()
        
    def __getitem__(self, index):
        return self.words[index] 
    
    def __len__(self):
        #completes sequence protocol, but not needed for iterable
        return len(self.words) 
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
```




```python
#sequence'
a= Sentence("Mary had a little lamb whose fleece was white as snow.")
len(a), a[3], a
```





    (11, 'little', Sentence('Mary had a l...hite as snow.'))





```python
min(a), max(a)
```





    ('Mary', 'whose')





```python
list(a)
```





    ['Mary',
     'had',
     'a',
     'little',
     'lamb',
     'whose',
     'fleece',
     'was',
     'white',
     'as',
     'snow.']



To iterate over an object x, python automatically calls `iter(x)`. An **iterable** is something which, when `iter` is called on it, returns an **iterator**.

(1) if `__iter__` is defined, calls that to implement an iterator.

(2) if not  `__getitem__` starting from index 0

(3) otherwise raise TypeError

Any Python sequence is iterable because they implement `__getitem__`. The standard sequences also implement `__iter__`; for future proofing you should too because  (2) might be deprecated in a future version of python.

This:



```python
for i in a:
    print(i)
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.


is implemented something like this:



```python
it = iter(a)
while True:
    try:
        nextval = next(it)
        print(nextval)
    except StopIteration:
        del it
        break
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.


`it` is an iterator. 

An iterator defines both `__iter__` and a `__next__` (the first one is only required to make sure an *iterator* IS an *iterable*). 

Calling `next` on an iterator will trigger the calling of `__next__`.



```python
it=iter(a)#an iterator defines `__iter__` and can thus be used as an iterable
for i in it:
    print(i)
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.




```python
it = iter(a)
next(it), next(it), next(it)
```





    ('Mary', 'had', 'a')



So now we can completely abstract away a sequence in favor an iterable (ie we dont need to support indexing anymore). From Fluent:



```python
class SentenceIterator:
    def __init__(self, words): 
        self.words = words 
        self.index = 0
        
    def __next__(self): 
        try:
            word = self.words[self.index] 
        except IndexError:
            raise StopIteration() 
        self.index += 1
        return word 

    def __iter__(self):
        return self
    
class Sentence:#an iterable
    def __init__(self, text): 
        self.text = text
        self.words = text.split()
        
    def __iter__(self):
        return SentenceIterator(self.words)
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
```




```python
s2 = Sentence("While we could have implemented `__next__` in Sentence itself, making it an iterator, we will run into the problem of exhausting an iterator'.")
```




```python
len(s2)
```



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-42-4b382c72e9ab> in <module>()
    ----> 1 len(s2)
    

    TypeError: object of type 'Sentence' has no len()




```python
for i in s2:
    print(i)
```


    While
    we
    could
    have
    implemented
    `__next__`
    in
    Sentence
    itself,
    making
    it
    an
    iterator,
    we
    will
    run
    into
    the
    problem
    of
    exhausting
    an
    iterator'.




```python
s2it=iter(s2)
print(next(s2it))
s2it2=iter(s2)
next(s2it),next(s2it2)
```


    While





    ('we', 'While')



While we could have implemented `__next__` in Sentence itself, making it an iterator, we will run into the problem of "exhausting an iterator". 

The iterator above keeps state in `self.index` and we must be able to start anew by creating a new instance if we want to re-iterate. Thus the `__iter__` in the iterable, simply returns the `SentenceIterator`.



```python
min(s2), max(s2)
```





    ('Sentence', 'will')



Note that min and max will work even though we now DO NOT satisfy the sequence protocol, but rather the ITERABLE protocol, as its a pairwise comparison, which can be handled via iteration. The take home message is that in programming with these iterators, these generlization of pointers, we dont need either the length or indexing to work to implement many algorithms: we have abstracted these away.

## Generators

EVERY collection in Python is iterable.

Lets pause to let that sink in.

We have already seen iterators are used to make for loops. They are also used tomake other collections

to loop over a file line by line from disk
in the making of list, dict, and set comprehensions
in unpacking tuples
in parameter unpacking in function calls (*args syntax)
An iterator defines both __iter__ and a __next__ (the first one is only required to make sure an iterator IS an iterable).

SO FAR: Iterator: retrieves items from a collection. The collection must implement __iter__.

### Yield and generators

A generator function looks like a normal function, but instead of returning values, it yields them. The syntax is (unfortunately) the same.

Unfortunate, as a generator is a different beast. When the function runs, it creates a generator.

The generator is an iterator.. It gets an internal implementation of __next__ and __iter__, almost magically.



```python
def gen123():
    print("Hi")
    yield 1
    print("Little")
    yield 2
    print("Baby")
    yield 3
```




```python
print(gen123, type(gen123))
g = gen123()
type(g)
```


    <function gen123 at 0x1067321e0> <class 'function'>





    generator





```python
#a generator is an iterator
g.__iter__
```





    <method-wrapper '__iter__' of generator object at 0x106728a98>





```python
g.__next__
```





    <method-wrapper '__next__' of generator object at 0x106728a98>





```python
next(g),next(g), next(g)
```


    Hi
    Little
    Baby





    (1, 2, 3)



When next is called on it, the function goes until the first yield. The function body is now suspended and the value in the yield is then passed to the calling scope as the outcome of the next.

When next is called again, it gets __next__ called again (implicitly) in the generator, and the next value is yielded..., and so on... ...until we reach the end of the function, the return of which creates a StopIteration in next.

Any Python function that has the yield keyword in its body is a generator function.



```python
for i in gen123():
    print(i)
```


    Hi
    1
    Little
    2
    Baby
    3


Use the language: "a generator yields or produces values"



```python
class Sentence:#an iterable
    def __init__(self, text): 
        self.text = text
        self.words = text.split()
        
    def __iter__(self):#one could also return iter(self.words)
        for w in self.words:#note this is implicitly making an iter from the list
            yield w
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
a=Sentence("Mary had a little lamb whose fleece was white as snow.")
```




```python
for w in a:
    print(w)
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.


## Lazy processing

Upto now, it might just seem that we have just represented existing sequences in a different fashion. But notice above, with the use of yield, that we do not have to define the entire sequence ahead of time. Indeed we talked about this a bit when we talked about iterators, but we can see this "lazy behavior" more explicitly now. We see it in the generation of infinite sequences, where there is no data per se!

So, because of generators, we can go from fetching items from a collection to "generate"ing iteration over arbitrary, possibly infinite series...



```python
def fibonacci(): 
    i,j=0,1 
    while True: 
        yield j
        i,j=j,i+j
```




```python
f = fibonacci()
for i in range(10):
    print(next(f))
```


    1
    1
    2
    3
    5
    8
    13
    21
    34
    55


## Lazy implementation for Sequences using generators¶

Despite all our talk of lazy implementation, our Sentence implementations so far have not been lazy because the init eagerly builds a list of all words in the text, binding it to the self.words attribute. This will entail processing the entire text, and the list may use as much memory as the text itself



```python
import re
WORD_REGEXP = re.compile('\w+')
class Sentence:#an iterable
    def __init__(self, text): 
        self.text = text
        
    def __iter__(self):
        for match in WORD_REGEXP.finditer(self.text):
            yield match.group()
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
```




```python
list(Sentence("the mad dog went home to his cat"))
```





    ['the', 'mad', 'dog', 'went', 'home', 'to', 'his', 'cat']



### Generator Expressions of data sequences.

There is an even simpler way: use a generator expression, which is just a lazy version of a list comprehension. (itrs really just sugar for a generator function, but its a nice bit of sugar)



```python
RE_WORD = re.compile('\w+')
class Sentence:#an iterable
    def __init__(self, text): 
        self.text = text
        
    def __iter__(self):
        return (match.group() for match in RE_WORD.finditer(self.text))
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
list(Sentence("the mad dog went home to his cat"))
```





    ['the', 'mad', 'dog', 'went', 'home', 'to', 'his', 'cat']



Which syntax to choose?

Write a generator function if the code takes more than 2 lines.

Some syntax that might trip you up: double brackets are not necessary



```python
(i*i for i in range(5))
```





    <generator object <genexpr> at 0x106728b48>





```python
list((i*i for i in range(5)))
```





    [0, 1, 4, 9, 16]





```python
list(i*i for i in range(5))
```





    [0, 1, 4, 9, 16]


