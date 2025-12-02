# Ways to handle dimensions

- Completely statically checked as actual ints
- Completely runtime checked
- Combinations

# Need for actual ints at compile time
- Broadcasting of pytorch needs to do comparisons between dimensions, so they can't just be opaque types.
- Concatenation of tensors at the least needs plus and minus arithmetic.

# Need for dimensions at runtime

## Vary a hyperparameter
Not really, just write the invoked method with a generic type parameter of the size being varied

## Reading of unknown size files
They're typically split into training examples of known explicit size fairly early on. Let's find examples.

## Reading unknown size of training examples
Also fine, we'll pick 1 or some other known batch afterwards.

# How to get dimensions at runtime
Ignore the types, just use `torch.size`.

