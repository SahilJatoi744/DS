# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(style="ticks", color_codes=True)

# titanic = sns.load_dataset("titanic")
# g = sns.FacetGrid(titanic, row="sex", hue="alone")
# g = (g.map(plt.scatter, "age", "fare").add_legend())
# plt.show()


# step 1   import libraries
# import seaborn as sns
# import matplotlib.pyplot as plt

# # step 2   set a theme
# sns.set_theme(style="ticks",color_codes=True)

# # step 3   import data set or you can also import your own data
# kashti = sns.load_dataset("titanic")
# # step 4   plot basic graph with out hue

# # p = sns.countplot(x="sex",data=kashti)
# # plt.show()

# # # step 4   plot basic graph with hue

# # p = sns.countplot(x="sex",data=kashti, hue="class")
# # plt.show()

# # step 4   plot basic graph with hue and title

# p = sns.countplot(x="sex",data=kashti, hue="class")
# p.set_title("Graph")
# plt.show()

# own data in graph

#importing libraries
import pandas as pd

#import data from files
chilla = pd.read_csv("filename.extension")
print(chilla)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)

p = sns.countplot(x="sex",data=chilla, hue="class")
p.set_title("Graph")
plt.show()
