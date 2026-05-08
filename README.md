# Quadratic NN test

This repo tests quadratic NNs where instead of $W*x+b$ we have $W_quad*x^2+W_lin+b$.

This project is intrnded both as learning and an experiment:
I was curious if this would be more powerful than a linear layer, and when I thought about it there were several already existing reasons why I'm trying:

## Proofs

1. Taylor Series Math

Linear approximation is $f(x+h) = f(x)+hf'(x)$

Quadratic approximation is $f(x+h) = f(x)+hf'(x) + h^2/2!*f''(x)$

As long as f''(x) isn't 0 the quadratic approximation is better. Also:

Linear error scales with the next term, h²/2!. Whereas quadratic error scales with the next term, h³/3!. So quadratic should be more accurate by a factor of $cuberoot(2!/3!)*h^2/3$ which is approximately 0.693*n_linear^2/3

Note however that is only about expressiveness of single neurons: when we assume a function with many inputs this breaks down as it requires the sum of all pairs. I'm not good enough at math to figure out how much better linear vs squarewise is in terms of error with multiple inputs but ye

But the point remains: quadratic nodes are more expressive than linears for curves as long as f''(x) exists: a quadratic could represent a curve in a few nodes that tskes linear tens of nodes.

#~ I'm just 14 messing around with taylor series, please tell me of any mistakes it just seems intuitive

2. Simpson's Rule

In integration, Simpson's rule using parabola gives a more accurate approximation of the integral. So I thought the same accuracy should apply here.

Simpson's rule is around sqrt(linear) better, though Simpson's Rule can also integrate cubics due to cool symmetry which is what makes it sqrt() not root[1.5]

Obviously integration is a bit unrelated but it does suggest that quadratic functions can be more accurate than linear ones. 

3. KANs

Kognorognov Arnold networks (i didn't spell that right) involve polynomials too, just with B-splines. Hlwever KANs are super hard to compute. I thought that if we added polynomials to the mix of standard NN we might get some of the same benefits with just one extra matmul.

Also, on a test on TinyStories and numpy, I managed to get a 120k quadratic to 2.65 validation loss (but due tk compute restraints I was unable to train on the full dataset. but no overfitting occurrd). For context the 1M linear transformer (official version is 3.7M dominated by embeddings) loss was 2.39. So it was quite promising. I'm now testing on BabyLM. The math suggests quadratic scales sublinearly so I'm quite hopeful (though again it could be catastrophically wrong)

4. Linears are a subset of quadratics

Linear equations are a subset of quadratics where the quadratic wright is 0. So in theory, a quadratic model should be able to learn to be linear if that's what the data requires. So it should be *at least* as good as a linear model, with options to use squared if that's what's needed.

In the future I could add skipping of zeroes to not have to compute quadratics if they aren't needed but I'm a bit lazy


## Challenges

1. Quadratic layers are more expensive to compute, meaning a quadratic model with half the layer  count of a linear model will actually have the same parameters.

2. Quadratics are more expressive but also that means they can oscillate and overfit (runge's phenomenon). Which is exactly why polynomial regression doesn't work. However, Runge's phenomenon mainly applies for high order polynomials not just a quadratic. Just in case I'm planning on careful L-n penalties and dropout which is already standard

3. Since to calculate $dOutput/dInput$ you calculate 2ax+b, not just b, the gradients will be more prone to expllosion and vanishing. So I need to add gradient clipping, dropout, careful initialization, label smoothing, etc.

4. Finally, the backprop is just more complex. So it's harder to write

## Notes

My plan is to make this *exactly* the same as a transformer just with squarewise FFN. So still some Re/GeLu, attention, etc.

