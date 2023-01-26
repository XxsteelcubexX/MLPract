import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.arange(-3,3,0.001)

def graph_simple():
    plt.plot(x, norm.pdf(x))
    plt.show()

def graph_multiple():
    plt.plot(x,norm.pdf(x))
    plt.plot(x, norm.pdf(x, 1.0, 0.5))
    plt.show()

def graph_save():
    plt.plot(x,norm.pdf(x))
    plt.plot(x, norm.pdf(x,1.0,0.5))
    path = '/home/piyush/Projects/MLPract/libBasics/'
    basicPath = 'outputs/fig1.png'
    plt.savefig(path + basicPath, format = 'png')

def graph_with_custom_axes():
    axes = plt.axes()
    axes.set_xlim([-5,5])
    axes.set_ylim([0, 1.0])
    axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.plot(x, norm.pdf(x))
    plt.plot(x, norm.pdf(x,1.0,0.5))
    plt.show()

def graph_grid():
    axes = plt.axes()
    axes.set_xlim([-5,5])
    axes.set_ylim([0, 1.0])
    axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    axes.grid()
    plt.plot(x, norm.pdf(x))
    plt.plot(x, norm.pdf(x,1.0,0.5))
    plt.show()

def graph_lineTypes_colors():
    axes = plt.axes()
    axes.set_xlim([-5,5])
    axes.set_ylim([0, 1.0])
    axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    axes.grid()
    plt.plot(x, norm.pdf(x), 'b--')
    plt.plot(x, norm.pdf(x,1.0,0.5), 'g:')
    plt.show()

def graph_labelingAxes_and_Legend():
    axes = plt.axes()
    axes.set_xlim([-5,5])
    axes.set_ylim([0, 1.0])
    axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    axes.grid()
    plt.xlabel('Greebles')
    plt.ylabel('Probability')
    plt.plot(x, norm.pdf(x), 'b--')
    plt.plot(x, norm.pdf(x,1.0,0.5), 'g:')
    plt.legend(['Sneetches','Gacks'], loc = 4)
    plt.show()

def easterEgg():
    plt.xkcd()
    axes = plt.axes()
    axes.set_xlim([-5,5])
    axes.set_ylim([0, 1.0])
    axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    axes.grid()
    plt.xlabel('Greebles')
    plt.ylabel('Probability')
    plt.plot(x, norm.pdf(x), 'b--')
    plt.plot(x, norm.pdf(x,1.0,0.5), 'g:')
    plt.legend(['Sneetches','Gacks'], loc = 4)
    plt.show()
    plt.rcdefaults()

def pieChart():
    values = [12,55,4,32,14]
    colours = ['r','g','b','c','m']
    explode = [0,0,0.2,0,0]
    labels = ['A','B','C','D','E']
    plt.pie(values, colors= colours, labels=labels, explode=explode)
    plt.title('Loactions of Different Food')
    plt.show()

def barChart():
    values = [12,55,4,32,14]
    colours = ['r','g','b','c','m']
    plt.bar(range(0,5), values, color = colours)
    plt.show()

def scatter_plot():
    x = np.random.randn(500)
    y = np.random.randn(500)
    plt.scatter(x,y)
    plt.show()

def histogram_plot():
    incomes = np.random.normal(27000, 15000, 10000)
    plt.hist(incomes, 50)
    plt.show()

def box_wisker_plot():
    uniformSkewed = np.random.rand(100)*100-40
    high_outliers = np.random.rand(10)*50+100
    low_outliers = np.random.rand(10)*(-50)-100
    data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
    plt.boxplot(data)
    plt.show()

def main():
    graph_simple()
    graph_multiple()
    graph_save()
    graph_with_custom_axes()
    graph_grid()
    graph_lineTypes_colors()
graph_labelingAxes_and_Legend()
easterEgg()
pieChart()
barChart()
scatter_plot()
histogram_plot()
box_wisker_plot()

if __name__ == '__main__':
    main()
