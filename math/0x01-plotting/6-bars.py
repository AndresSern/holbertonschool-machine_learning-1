#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
xs = np.arange(3)
plt.xticks(xs, ('Farrah', 'Fred', 'Felicia'))
fr = fruit
apples = plt.bar(xs, fr[0], 0.5, color='red')
bananas = plt.bar(xs, fr[1], 0.5, color='yellow', bottom=fr[0])
oranges = plt.bar(xs, fr[2], 0.5, color='#ff8000', bottom=fr[0]+fr[1])
peach = plt.bar(xs, fr[3], 0.5, color='#ffe5b4', bottom=fr[0] + fr[1]+ fr[2])
plt.legend((apples, bananas, oranges, peach),
           ('apples', 'bananas', 'oranges', 'peaches'))

plt.show()
