#             #Tuples

# # Ordered collection of elements
# # enclosed in ()
# # Different kind of elements are stored
# # Once elements are stored cannot be changed means unmutable

# tupl = (1, "Python", True, 2.6)
# print(tupl)

#             # Indexing in tuple

# print(tupl[0])
# print(tupl[1])
# print(tupl[2])
# print(tupl[3])


# # last element is exclusive
# print(tupl[0:3])

# tupl2 = (2 , "Sahil" , 2.5, False)

# print(tupl + tupl2)

# tupl3 = (20,30,40,50,60,70)

# print(tupl3)
# # it print tupl3 two times
# print(tupl3*2)
# # we cant apply arthmetric operations on tuple cuz we cant edit it or it is unmutable








#                     #Lists
# ordered collection of data
# enclosed in []
# mutable or can be changed

list1 = [2,"Sahil",False]
print(list1)
print(type(list1))
print(list1[1])

list2 = [3,5,"Sahil","Jatoi",478,52,8,False]

print(list1 + list2)

print(list1*2)

print(list1.reverse())

list1.append("Youtube")
print(list1)

print(list1.count("Y"))


list3 = [20,30,40,50,60,70]
print(list3)
print(len(list3))

list3.sort()
print(list3)

# to append to lists
print(list1 + list2)






#             #Dictionaries

#An unordered of collection of elements
# key and value
# enclosed in {} or ()
# Mutable/Changes the values

#Food and their prices

d1 = {"Samosa": 30,"Pakora":100,"Raita":30,"salad":50,"Chicken Rolls": 30}
print(d1)
print(type(d1)) 

# Extract data

keys = d1.keys()
print(keys)

values1 = d1.values()
print(values1)

# adding new elements

d1["Tikka"] =10
print(d1)

d1["Tikka"] =15
print(d1)

        # d2

d2 = {"Dates":50,"Choclates":200,"Sawaiyya":1000}
print(d2)

#concationate
d1.update(d2)









                #Sets


# unordered and unindexed 
# enclosed in {}
# No duplicates are allowed

s1 = {1,2,3,4,5,"Sahil","Alka","Shoaib"}
print(s1)

s1.add("Sajan")
print(s1)

s1.remove("Sajan")
print(s1)

