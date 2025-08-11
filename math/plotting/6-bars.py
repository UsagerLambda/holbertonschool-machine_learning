#!/usr/bin/env python3
'''
Display a bar graph that display the number of fruits each person have
'''
import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''
    Displays a stacked bar chart of fruit quantities for three people.
    '''
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))  # Tableau numpy 4 lines x 3 cols
    plt.figure(figsize=(6.4, 4.8))

    x = ['Farrah', 'Fred', 'Felicia']

    # Commence à 0
    plt.bar(x, fruit[0], color='red', width=0.5, label='Apples')

    # Commence au sommet de la première (bottom = valeur de fruit[0])
    plt.bar(x, fruit[1], bottom=fruit[0],
            color='yellow', width=0.5, label='Bananas')

    # Commence au sommet des deux autres (v de fruit[0] + fruit[1])
    plt.bar(x, fruit[2], bottom=fruit[0] + fruit[1],
            color='#ff8000', width=0.5, label='Oranges')

    # Commence au sommet des trois autres (v de fruit[0] + fruit[1] + fruit[2])
    plt.bar(x, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],
            color='#ffe5b4', width=0.5, label='Peaches')

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')

    plt.yticks(range(0, 81, 10))
    plt.ylim(0, 80)

    plt.legend()
    plt.show()
