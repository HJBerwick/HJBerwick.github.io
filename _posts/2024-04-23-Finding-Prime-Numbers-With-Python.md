---
layout: post
title: Finding Prime Numbers with Python
image: "/posts/primes_header_new.png"
tags: [Python, Primes]
---

In this post, I'm going to discuss how we can create a function in Python that, when passed a single number as an upper bound, will quickly calculate all the prime numbers up to this limit.

For anyone who is not familiar with prime numbers, these can be defined as those numbers which are wholly divisible only by themselves and 1. For example, 5 is a prime number as none of 2, 3 or 4 divide into 5 exactly (i.e. without remainder). However, 6 is not a prime number as both 2 and 3 divide into 6, rather than just 1 and itself.

---

To kick off, we're going to start small to understand the basic approach, before we then build out our function to handle much larger upper limits.

With that in mind, we will set up a variable to act as our upper bound, and will initially set this to 20, i.e. we will look to find all prime numbers below 20.

```python
n = 20
```

Now that we have our limit, we need to create a range of integers below our limit to check if they are prime numbers or not. We will start our range at 2, as this is the smallest prime number, and use n+1 as our stopping point as the range logic is not inclusive of the upper bound.

We will also be using a set, rather than a list, to hold these values as there are some important methods available to us with sets that will help us to efficiently update our list of potential primes as we go.

```python
number_range = set(range(2, n+1))
```

Now that we have our range of values to check, we also need to create somewhere to store any of the primes that we find, and we can use a list for this purpose.

```python
primes_list = []
```

Let's now start to code up the logic we can use to check for prime numbers within our range of integers. Later on we will be able to use this logic to build out our while loop to iterate through all possible values, but for now we can iterate manually to make sure this works for the simplest cases. This also allows us to iron out any issues before we let this run through everything on its own where any errors may be harder to understand.

To start with, we want to extract the first value from our set of numbers (called **number_range**) and check whether this is prime. If so, we can add this to our list of prime numbers (handily named **primes_list**), but if not, we can then discard this number.

```python
print(number_range)
>> {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```
The *pop* method extracts an element from a list or set and provides this value to us. If we use *pop*, this will remove the first element from our **number_range** set, and we can then assign this to an object called **prime**.

```python
prime = number_range.pop()
print(prime)
>> 2
print(number_range)
>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```

Given that we know the first value in our set is the smallest, we also know that there are no values, other than itself and 1, which will divide into this number and so we can safely say that this is a prime number. As a result, let's add this to our list of primes.

```python
primes_list.append(prime)
print(primes_list)
>> [2]
```

Now for our previous logic to continue to hold, i.e. that the lowest value in our range is prime, we need to make some adjustments. On our first iteration, our range started at 2, which we knew to be prime, and we also knew that there were no smaller divisors, other than 1, that existed outside of this set of numbers. As we have removed a value, to ensure the smallest value in our **number_range** set is prime, we need to check which numbers can be divided by the prime value we just found.

To do this, we are first going to calculate all multiples of our prime value under our upper limit. We will then be able to use this to compare with the remaining values in our **number_range** and remove the matching values. To help us with this, we're once again going to use a set here, as this will enable us to use some key functionality that is vital to this approach.

```python
multiples = set(range(prime*2, n+1, prime))
```

For our multiples, we again use the range functionality, which uses the syntax range(start, stop, step). For the purposes of comparison, we have already removed our prime value, so we can start at the next multiple, hence **prime** * 2. We also know that our range functionality is not inclusive of our upper bound so we will, once again, use n+1 here to ensure our limit is included in the comparison. Finally, as we are looking at multiples of our prime value, this will also be our step value.

We can now take a look at our list of multiples:

```python
print(multiples)
>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

This brings us to the key functionality we referenced above, which really underpins our approach.This functionality is the **difference_update** method for sets which will remove any elements from our set that are also found to be present in a comparison set. For our purposes, this means we can remove any values from our **number_range** set which are also listed as **multiples** of our previously found prime number. Given that these multiples are divisible by more than just themselves and 1, we know they are not prime and so we can happily discard these values.

Before we apply the **difference_update**, we can visually compare our two sets.

```python
print(number_range)
>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

print(multiples)
>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

To use **difference_update**, we can put our **number_range** set and then apply the difference update with respect to our **multiples** to only include the values that are *different* from those in a second set.

```python
number_range.difference_update(multiples)
print(number_range)
>> {3, 5, 7, 9, 11, 13, 15, 17, 19}
```

We can now see that our list of potential prime numbers has decreased significantly. In fact, every time we find a new prime, we will also significantly reduce our list of potentials by calculating further multiples, leading to a smaller and smaller pool of data points to assess, and hence this turns out to be a really efficient approach.

By removing the previous prime number and all of its multiples, we are now back to a position where we know that the smallest value in our **number_range** is also prime, and thus we are back to our starting point. This means we are in a position to iterate through the remaining values in a similar way.

For the small range below 20 that we have been considering, we could quite quickly iterate through the remaining values manually. However, for much larger upper limits we are going to want to automate this iteration, and thus we will now look to apply our logic within a while loop to help us with this.

We can see below how we could apply this approach if we increase our upper limit to 1000, with the while loop iterating through all possible primes until our **number_range** is empty:

```python
n = 1000

# number range to be checked
number_range = set(range(2, n+1))

# empty list to append discovered primes to
primes_list = []

# iterate until list is empty
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
```

We can then print our list of primes to see what we have come up with.

```python
print(primes_list)
>> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
```

As we can see above, when we begin to use larger upper limits, we will start to see a lot of data returned. To help us get our head around what we've found, we could also explore some interesting summary statistics for our list of primes. For example, we can take a look at the total number of primes found, or the largest in our list.

```python
prime_count = len(primes_list)
largest_prime = max(primes_list)
print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
>> There are 168 prime numbers between 1 and 1000, the largest of which is 997
```

Now that we have successfully created our method to find primes up to a given limit, and we have a good way to summarise our findings, all that is left is to add this logic within a function so we can easily use this in future.

```python
def primes_finder(n):
    
    # number range to be checked
    number_range = set(range(2, n+1))

    # empty list to append discovered primes to
    primes_list = []

    # iterate until list is empty
    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
        
    prime_count = len(primes_list)
    largest_prime = max(primes_list)
    print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
```

With this function in place, we can now pass it any upper bound we wish, and the function will do the rest for us. We can now even go for a much larger value. For example, let's try an upper bound of 1,000,000.

```python
primes_finder(1000000)
>> There are 78498 prime numbers between 1 and 1000000, the largest of which is 999983
```

That's quite a change of gear from our initial limit of 20!

I hope you enjoyed following along on this search for prime numbers with Python.

---

###### Important Note: Using pop() on a Set in Python

We used the pop() method in our solution above, however, in practice this can lead to some inconsistencies when used on a Set.

While the elements of a Set are stored internally with some order (determined by the hash code of the key), Sets are, by definition, unordered. This means that while the pop() method will usually extract the lowest element of a set, we cannot 100% rely on this being the case. The hashing method (which also allows for such fast retrieval) can mean that, in rare cases, the hash does not provide the lowest value.

If we were using Sets and pop() in Python in the future, we might want to implement a slight adjustment, such that the line below:

```python
prime = number_range.pop()
```

Could be replaced with the following code:

```python
prime = min(sorted(number_range))
number_range.remove(prime)
```

This effectively splits the pop functionality into two steps. Firstly, we ensure the correct identification of the lowest value in our **number_range** by taking the minimum of our sorted Set. We will still assign this value to our **prime** object. However, we then perform a second, separate step to actually remove this value from our list of potential primes.

Due to the repeated sorting that takes place in each iteration of our while loop, this alternative is slightly slower than our original solution using pop()!

