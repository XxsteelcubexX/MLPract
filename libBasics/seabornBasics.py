%matplotlib inline
import pandas as pd
import seaborn as sns
path = '~/Projects/MLPract'
fileName = '/dataSets/FuelEfficiency.csv'
df = pd.read_csv(path+fileName)

def barChart():
    gear_counts = df['# Gears'].value_counts()
    gear_counts.plot(kind = 'bar')

def seaborn_bar_chart():
    gear_counts = df['# Gears'].value_counts()
    sns.set()
    gear_counts.plot(kind = 'bar')

def seaborn_distPlot():
    # there is no such plot in matplotlib
    sns.displot(df['CombMPG'])

def pair_plots():
    df2 = df[['Cylinders','CityMPG','HwyMPG','CombMPG']]
    df2.head()
    sns.pairplot(df2,height=2.5,hue=None)

def scatter_plot():
    sns.scatterplot(x='Eng Displ',y='CombMPG', data = df)

def JoinPlots():
    sns.jointplot(x = 'Eng Displ', y='CombMPG', data = df)

def scatter_with_regression_line():
    sns.lmplot(x = 'Eng Displ', y='CombMPG', data = df)

def box_plot():
    sns.set(rc={'figure.figsize':(15,5)})
    ax = sns.boxplot(x='Mfr Name', y = 'CombMPG',data =df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation =90)
    #dir(ax)

def swarm_plot():
    ax = sns.swarmplot(x='Mfr Name', y = 'CombMPG',data =df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation =90)

def count_plot():
    ax = sns.countplot(x = 'Mfr Name', data = df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation =90)

def heatMap():
    df2 = df.pivot_table(index = 'Cylinders', columns='Eng Displ', values = 'CombMPG', aggfunc = 'mean')
    sns.heatmap(df2)

def main():
    barChart()
    seaborn_bar_chart()
df.head()
seaborn_distPlot()
pair_plots()
scatter_plot()

# seaborn also offers a "joinplot". which combines a scatterplot with histogram on both axes.
# This lets you Visualize both the individual data points and the distribution across both
# dimensions at the same time.
JoinPlots()

scatter_with_regression_line()

box_plot()

swarm_plot()
count_plot()
heatMap()

if __name__ == '__main__':
    main()
