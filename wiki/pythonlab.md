---
title: 
shorttitle: pythonlab
notebook: pythonlab.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}
## An introduction to Objects and their Classes in python



```python
from IPython.display import HTML
```


### Means of Abstraction: how to build complex units

We want to find a way to represent data in the context of our programming language. In particular, we are concerned with complex data, structured data. For example, to prepresent a location, we might want to associate a `name` with it, a `latitude`, and a `longitude`. Thus we would want to create a **compound data type** which carries this information. In C, for example, this is a struct:

```C
struct location {
    float longitude;
    float latitude;
}
```

- When we write a function, we give it some sensible name which can then be used by a "client" programmer. We dont care about how this function is implemented, but rather, just want to know its signature (API) and use it.

- In a similar way, we want to *encapsulate* our data: we dont want to know how it is stored and all that, but rather, just be able to use it. This is one of the key ideas behind object oriented programming. 

- To do this, write **constructors** that make objects, and other functions that access or change data on the object. These functions are called the "methods" of the object, and are what the client programmer uses.

### Python Classes and instance variables

Classes allow us to define our own *types* in the python type system. 



```python
class ComplexClass():
    
    def __init__(self, a, b):
        self.real = a
        self.imaginary = b

```




```python
c1 = ComplexClass(1,2)
print(c1, c1.real)
```


    <__main__.ComplexClass object at 0x1073d9ba8> 1




```python
vars(c1), type(c1)
```





    ({'imaginary': 2, 'real': 1}, __main__.ComplexClass)





```python
c1.real=5
print(c1, c1.real, c1.imaginary)
```


    <__main__.ComplexClass object at 0x1073d9ba8> 5 2


### Inheritance and Polymorphism

**Inheritance** is the idea that a "Cat" is-a "Animal" and a "Dog" is-a "Animal". "Animal"s make sounds, but Cats Meow and Dogs Bark. Inheritance makes sure that *methods not defined in a child are found and used from a parent*.

**Polymorphism** is the idea that an **interface** is specified (not necessarily implemented) by a superclass, and then its implemented in subclasses (differently). 



```python
class Animal():
    
    def __init__(self, name):
        self.name = name
        
    def make_sound(self):
        raise NotImplementedError
    
class Dog(Animal):
    
    def make_sound(self):
        return "Bark"
    
class Cat(Animal):
    
    def __init__(self, name):
        self.name = "Best Animal %s" % name
        
    def make_sound(self):
        return "Meow"  
    
    
```




```python
a0 = Animal("Rahul")
print(a0.name)
a0.make_sound()
```


    Rahul



    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-20-18721729352b> in <module>()
          1 a0 = Animal("Rahul")
          2 print(a0.name)
    ----> 3 a0.make_sound()
    

    <ipython-input-19-57b210a55e9d> in make_sound(self)
          5 
          6     def make_sound(self):
    ----> 7         raise NotImplementedError
          8 
          9 class Dog(Animal):


    NotImplementedError: 




```python
a1 = Dog("Snoopy")
a2 = Cat("Tom")
animals = [a1, a2]
for a in animals:
    print(a.name)
    print(isinstance(a, Animal))
    print(a.make_sound())
    print('--------')
```


    Snoopy
    True
    Bark
    --------
    Best Animal Tom
    True
    Meow
    --------




```python
print(a1.make_sound, Dog.make_sound)
```


    <bound method Dog.make_sound of <__main__.Dog object at 0x1073e6588>> <function Dog.make_sound at 0x1073d3bf8>




```python
print(a1.make_sound())
print('----')
print(Dog.make_sound(a1))
```


    Bark
    ----
    Bark




```python
Dog.make_sound()
```



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-24-11ebae4e7564> in <module>()
    ----> 1 Dog.make_sound()
    

    TypeError: make_sound() missing 1 required positional argument: 'self'


### Calling a superclasses initializer

Say we dont want to do all the work of setting the name variable in the subclasses. We can set this "common" work up in the superclass and use `super` to call the superclass'es initializer from the subclass (See https://rhettinger.wordpress.com/2011/05/26/super-considered-super/)



```python
class Animal():
    
    def __init__(self, name):
        self.name=name
        print("Name is", self.name)


        
class Mouse(Animal):
    def __init__(self, name):
        self.animaltype="prey"
        super().__init__(name)
        print("Created %s as %s" % (self.name, self.animaltype))
    
class Cat(Animal):
    pass

a1 = Mouse("Tom")
print(vars(a1))
a2 = Cat("Jerry")
print(vars(a2))
```


    Name is Tom
    Created Tom as prey
    {'animaltype': 'prey', 'name': 'Tom'}
    Name is Jerry
    {'name': 'Jerry'}


### Interfaces

The above examples show inheritance and polymorphism. But notice that we didnt actually need to set up the inheritance. We could have just defined 2 different classes and have them both `make_sound`, the same code would work. In java and C++ this is done more formally through Interfaces and  Abstract Base Classes respectively plus inheritance, but in Python this agreement to define `make_sound` is called "duck typing"



```python
#both implement the "Animal" Protocol, which consists of the one make_sound function
class Dog():
    
    def make_sound(self):
        return "Bark"
    
class Cat():
    
    def make_sound(self):
        return "Meow"  
    
a1 = Dog()
a2 = Cat()
animals = [a1, a2]
for a in animals:
    print(isinstance(a, Animal))
    print(a.make_sound())
```


    False
    Bark
    False
    Meow


### The Python Data Model

Duck typing is used throught python. Indeed its what enables the "Python Data Model" 

- All python classes implicitly inherit from the root **object** class.
- The Pythonic way, is to just document your interface and implement it. 
- This usage of common **interfaces** is pervasive in *dunder* functions to comprise the python data model.

####   `__repr__`  

The way printing works is that Python wants classes to implement a `__repr__` and a `__str__` method. It will use inheritance to give the built-in `object`s methods when these are not defined...but any class can define these. When an *instance* of such a class is interrogated with the `repr` or `str` function, then these underlying methods are called.

We'll see `__repr__` here. If you define `__repr__` you have made an object sensibly printable...



```python
class Animal():
    
    def __init__(self, name):
        self.name=name
        
    def __repr__(self):
        class_name = type(self).__name__
        return "Da %s(name=%r)" % (class_name, self.name)
```




```python
r = Animal("Rahul")
r
```





    Da Animal(name='Rahul')





```python
print(r)
```


    Da Animal(name='Rahul')




```python
repr(r)
```





    "Da Animal(name='Rahul')"



### The pattern with dunder methods


**there are functions without double-underscores that cause the methods with the double-underscores to be called**

Thus `repr(an_object)` will cause `an_object.__repr__()` to be called. 

In user-level code, you *SHOULD NEVER* see the latter. In library level code, you might see the latter. The definition of the class is considered library level code.

#### Instance Equality via `__eq__`

Now we are in a position to answer the initial question: what makes two squirrels equal!

To do  this, we will add a new dunder method to the mix, the unimaginatively (thats a good thing) named `__eq__`.



```python
class Animal():
    
    def __init__(self, name):
        self.name=name
        
    def __repr__(self):
        class_name = type(self).__name__
        return "%s(name=%r)" % (class_name, self.name)
    
    def __eq__(self, other):
        return self.name==other.name # two animals are equal if there names are equal
```




```python
A=Animal("Tom")
B=Animal("Jane")
C=Animal("Tom")
```


Three separate object identities, but we made two of them equal!



```python
print(id(A), id(B), id(C))

print(A==B, B==C, A==C)
```


    4416444736 4416444848 4416445856
    False False True


This is critical because it gives us a say in what equality means

### Python's power comes from the data model, composition, and delegation

The data model is used (from Fluent) to provide a:

>description of the interfaces of the building blocks of the language itself, such as sequences, iterators, functions, classes....

The special "dunder" methods we talk about are invoked by the python interpreter to beform basic operations. For example, `__getitem__` gets an item in a sequence. This is used to do something like `a[3]`. `__len__` is used to say how long a sequence is. Its invoked by the `len` built in function. 

A **sequence**, for example,  must implement `__len__` and `__getitem__`. Thats it.

The original reference for this data mode is: https://docs.python.org/3/reference/datamodel.html .

### Building out our class: instances and classmethods



```python
class ComplexClass():
    def __init__(self, a, b):
        self.real = a
        self.imaginary = b
        
    @classmethod
    def make_complex(cls, a, b):
        return cls(a, b)
        
    def __repr__(self):
        class_name = type(self).__name__
        return "%s(real=%r, imaginary=%r)" % (class_name, self.real, self.imaginary)
        
    def __eq__(self, other):
        return (self.real == other.real) and (self.imaginary == other.imaginary)
```




```python
c1 = ComplexClass(1,2)
c1
```





    ComplexClass(real=1, imaginary=2)



`make_complex` is a class method. See how its signature is different above. It is a factory to produce instances.



```python
c2 = ComplexClass.make_complex(1,2)
c2
```





    ComplexClass(real=1, imaginary=2)





```python
c1 == c2
```





    True



You can see where we are going with this. Wouldnt it be great to define adds, subtracts, etc? Later...

### Class variables and instance variables





```python
class Demo():
    classvar=1
      
ademo = Demo()
print(Demo.classvar, ademo.classvar)
ademo.classvar=2 #different from the classvar above
print(Demo.classvar, ademo.classvar)
```


    1 1
    1 2

