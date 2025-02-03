import matplotlib.pyplot as plt

plt.style.use('ggplot')

dic = {'Class a': 26.9,
       'Class b': 18,
       'Class c': 16.8,
       'Class d': 13,
       'Class e': 8.83,
       'Class f': 5.97,
       'Class g': 3.59,
       'Class h': 2.01,
       'Class i': 1.42,
       'Class j': 1.09,
       'Class k': 0.903,
       'Class l': 0.873,
       'Class m': 0.28,
       'Class n': 0.24,
       'Class o': 0.112}

# group together all elements in the dictionary whose value is less than 2
# name this group 'All the rest'
import itertools
newdic={}
for key, group in itertools.groupby(dic, lambda k: 'All the rest' if (dic[k]<2) else k):
     newdic[key] = sum([dic[k] for k in list(group)])   

labels = newdic.keys()
sizes = newdic.values()

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, autopct='%1.1f%%', explode=(0,0,0,0,0,0,0,0,0), startangle=0)
ax.axis('equal')
plt.tight_layout()

plt.show()