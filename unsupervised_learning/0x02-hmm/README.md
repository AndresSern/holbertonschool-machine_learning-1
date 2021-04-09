# 0x02. Hidden Markov Models
Hidden Markov models are generative models, in which the joint distribution of observations and hidden states, or equivalently both the prior distribution of hidden states (the transition probabilities) and conditional distribution of observations given states (the emission probabilities), is modeled. 
## 0. Markov Chain
``` python
#P :the transition matrix
#S: representing the probability of starting in each state
S[i] = P * S[i-1]
```
## 1. Regular Chains
```python
s * p = s
```
## 2. Absorbing Chains
In the mathematical theory of probability, an absorbing Markov chain is a Markov chain in which every state can reach an absorbing state. An absorbing state is a state that, once entered, cannot be left.
## 3. The Forward Algorithm
![alt text](https://i.ibb.co/rp58J8X/forward.png)
## 4. The Viretbi Algorithm
![alt text](https://i.ibb.co/d2QTTr2/viterbi.png)