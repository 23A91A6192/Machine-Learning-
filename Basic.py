import pandas as pd
data=pd.read_csv('iris.csv')
#print(data) 
#print(data.head(10)) #to print 10 rowws of data starting with 0
#print(data.describe()) #to get complete description like mean, min,max
#print(data.info()) #to get information about data like non-null values,data type,memory usage
#print(data.dtypes) #to get the data type of each column
#print(data.shape) #to get the number of rows and columns
#print(data.values) #to get the values of the data
#print(data.sort_values('SW',ascending=True)) #to sort the data of SW column in ascending order 
#print(data.sort_index(ascending=True)) #to sort the index
#print(data.sample(10)) #to get random rows from the data
#print(data.nlargest(2,'SW')) #to get the largest values in the column
#print(data.nsmallest(2,'SW')) #to get the 2 smallest values in the paritcular column
#print(data[data.PW>=2.0]) #compare the values of that  column and gives the value
#print(data['SW']) #to print the particular column
#print(data[['SW','SL']]) #calling multiple columns
#print(data.loc[0:9,'SL':'PW']) #to get the values from 0 to 9 and from SL to PW
#print(data.loc[data['SL']>5.0,['SW','SL']]) #all the SL values greater than 5 prints the SW values
#print(data.iloc[2:5])#index location where we get the matrix from index 2 to 5
#print(data.loc[22]) #to check particular row
#print(data.iloc[:,2]) #print all rows at 2nd column
print(data.iloc[2,2]) #print the value at 2nd row and 2nd column
print(data.SW.unique()) #returns all the unique values
print(data.SW.value_counts()) #counts the number of the times unique value repeated
print(data['SW'].sum()) #the sum of the values in the SW column
print(data['SW'].min()) #prints the minimum value in the SW column
print(data['SW'].max()) #prints the maximum value in the SW column
print(data['SW'].median()) #print the median value in the SW column