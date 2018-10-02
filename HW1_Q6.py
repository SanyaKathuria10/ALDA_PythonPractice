#Authors: Aayush Patial (apatial) , Rajat Narang (rnarang) , Sanya Kathuria (skathur2)
import os
import pandas
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import scipy



# Importing the Data file
# path =  os.getcwd()
# data = pandas.read_csv(path + '/Data/hw1q6_data.csv')

#Change the path here if needed
data = pandas.read_csv('Data/hw1q6_data.csv')

#part A

#Searching for 1 and 0 values in the last column for Diabetes detection
diab = 0
nondiab = 0
for row in data.iloc[:,-1]:
    if row == 1:
        diab +=1
    elif row == 0:
        nondiab +=1


print('\n')
print('Part A')
print(' Number of Diabetic Patients : ' + str(diab))
print(' Number of Nondiabetic Patients : ' + str(nondiab))
print('\n')



#part B

#Counting for missing values marked 0 in each coloumn
count = 0
for row in data.iloc[:,0]:
    if row == 0:
        count +=1

#Diving the counted values by total values and multiplying with 100 to get percentage
Percentage_Missing_Glucose = (count/768)*100

count = 0
for row in data.iloc[:,1]:
    if row == 0:
        count +=1
Percentage_Missing_BP = (count/768)*100

count = 0
for row in data.iloc[:,2]:
    if row == 0:
        count +=1
Percentage_Missing_ST = (count/768)*100

count = 0
for row in data.iloc[:,3]:
    if row == 0:
        count +=1
Percentage_Missing_BMI = (count/768)*100

count = 0
for row in data.iloc[:,4]:
    if row == 0:
        count +=1
Percentage_Missing_DPF = (count/768)*100

count = 0
for row in data.iloc[:,5]:
    if row == 0:
        count +=1
Percentage_Missing_Age = (count/768)*100

print('Part B')
print(' Percentage of missing data for Glucose : ' + str(Percentage_Missing_Glucose))
print(' Percentage of missing data for BloodPressure : ' + str(Percentage_Missing_BP))
print(' Percentage of missing data for SkinThickness : ' + str(Percentage_Missing_ST))
print(' Percentage of missing data for BMI : ' + str(Percentage_Missing_BMI))
print(' Percentage of missing data for DiabetesPedigreeFunction : ' + str(Percentage_Missing_DPF))
print(' Percentage of missing data for Age : ' + str(Percentage_Missing_Age))
print('\n')




#Part C

#Dropping the rows with missing values i.e. with 0 value for any attributes
for index, row in data.iloc[:,0:6].iterrows():
    if row['Glucose'] == 0 or row['BloodPressure'] == 0 or row['SkinThickness'] == 0 or row['BMI'] == 0 or row['DiabetesPedigreeFunction']==0 or row['Age'] == 0:
        data.drop(index,inplace = True)



#Part D
#Searching for 1 and 0 values in the last column for Diabetes detection in remaining data
diab_new = 0
nondiab_new = 0
for row in data.iloc[:,-1]:
    if row == 1:
        diab_new +=1
    elif row == 0:
        nondiab_new +=1

print('Part D')
print(' After Removing Patients(rows) with missing values ')
print('\n')
print(' Number of Diabetic Patients in remaining data : ' + str(diab_new))
print(' Number of Nondiabetic Patients in remaining data : ' + str(nondiab_new))
print('\n')


#Part E
# Finding Mean, Median, Standard Deviation, Range, 25th percentiles, and 50th percentiles, 75th percentiles for each feature
mean_Glucose = data['Glucose'].mean()
median_Glucose = data['Glucose'].median()
std_Glucose = data['Glucose'].std()
range_Glucose = [data['Glucose'].min(),data['Glucose'].max()]
TwentyFivePer_Glucose = data['Glucose'].quantile(0.25)
FiftyPer_Glucose = data['Glucose'].quantile(0.5)
SeventyFivePer_Glucose = data['Glucose'].quantile(0.75)

mean_BP = data['BloodPressure'].mean()
median_BP = data['BloodPressure'].median()
std_BP = data['BloodPressure'].std()
range_BP = [data['BloodPressure'].min(),data['BloodPressure'].max()]
TwentyFivePer_BP = data['BloodPressure'].quantile(0.25)
FiftyPer_BP = data['BloodPressure'].quantile(0.5)
SeventyFivePer_BP = data['BloodPressure'].quantile(0.75)

mean_ST = data['SkinThickness'].mean()
median_ST = data['SkinThickness'].median()
std_ST = data['SkinThickness'].std()
range_ST = [data['SkinThickness'].min(),data['SkinThickness'].max()]
TwentyFivePer_ST = data['SkinThickness'].quantile(0.25)
FiftyPer_ST = data['SkinThickness'].quantile(0.5)
SeventyFivePer_ST = data['SkinThickness'].quantile(0.75)

mean_BMI = data['BMI'].mean()
median_BMI = data['BMI'].median()
std_BMI = data['BMI'].std()
range_BMI = [data['BMI'].min(),data['BMI'].max()]
TwentyFivePer_BMI = data['BMI'].quantile(0.25)
FiftyPer_BMI = data['BMI'].quantile(0.5)
SeventyFivePer_BMI = data['BMI'].quantile(0.75)

mean_DPF = data['DiabetesPedigreeFunction'].mean()
median_DPF = data['DiabetesPedigreeFunction'].median()
std_DPF = data['DiabetesPedigreeFunction'].std()
range_DPF = [data['DiabetesPedigreeFunction'].min(),data['DiabetesPedigreeFunction'].max()]
TwentyFivePer_DPF = data['DiabetesPedigreeFunction'].quantile(0.25)
FiftyPer_DPF = data['DiabetesPedigreeFunction'].quantile(0.5)
SeventyFivePer_DPF = data['DiabetesPedigreeFunction'].quantile(0.75)

mean_Age = data['Age'].mean()
median_Age = data['Age'].median()
std_Age = data['Age'].std()
range_Age = [data['Age'].min(),data['Age'].max()]
TwentyFivePer_Age = data['Age'].quantile(0.25)
FiftyPer_Age = data['Age'].quantile(0.5)
SeventyFivePer_Age = data['Age'].quantile(0.75)

mean_Class = data['Class'].mean()
median_Class = data['Class'].median()
std_Class = data['Class'].std()
range_Class = [data['Class'].min(),data['Class'].max()]
TwentyFivePer_Class = data['Class'].quantile(0.25)
FiftyPer_Class = data['Class'].quantile(0.5)
SeventyFivePer_Class = data['Class'].quantile(0.75)



print('Part E')
print(' STATS FOR GLUCOSE')
print(' Glucose mean : ' + str(mean_Glucose))
print(' Glucose median : ' + str(median_Glucose))
print(' Glucose Standard Deviation : ' + str(std_Glucose))
print(' Glucose Range : ' + str(range_Glucose))
print(' Glucose 25th percentiles : ' + str(TwentyFivePer_Glucose))
print(' Glucose 50th percentiles : ' + str(FiftyPer_Glucose))
print(' Glucose 75th percentiles : ' + str(SeventyFivePer_Glucose))
print('\n')

print(' STATS FOR BLOODPRESSURE')
print(' BloodPressure mean : ' + str(mean_BP))
print(' BloodPressure median : ' + str(median_BP))
print(' BloodPressure Standard Deviation : ' + str(std_BP))
print(' BloodPressure Range : ' + str(range_BP))
print(' BloodPressure 25th percentiles : ' + str(TwentyFivePer_BP))
print(' BloodPressure 50th percentiles : ' + str(FiftyPer_BP))
print(' BloodPressure 75th percentiles : ' + str(SeventyFivePer_BP))
print('\n')

print(' STATS FOR SKINTHICKNESS')
print(' SkinThickness mean : ' + str(mean_ST))
print(' SkinThickness median : ' + str(median_ST))
print(' SkinThickness Standard Deviation : ' + str(std_ST))
print(' SkinThickness Range : ' + str(range_ST))
print(' SkinThickness 25th percentiles : ' + str(TwentyFivePer_ST))
print(' SkinThickness 50th percentiles : ' + str(FiftyPer_ST))
print(' SkinThickness 75th percentiles : ' + str(SeventyFivePer_ST))
print('\n')

print(' STATS FOR BMI')
print(' BMI mean : ' + str(mean_BMI))
print(' BMI median : ' + str(median_BMI))
print(' BMI Standard Deviation : ' + str(std_BMI))
print(' BMI Range : ' + str(range_BMI))
print(' BMI 25th percentiles : ' + str(TwentyFivePer_BMI))
print(' BMI 50th percentiles : ' + str(FiftyPer_BMI))
print(' BMI 75th percentiles : ' + str(SeventyFivePer_BMI))
print('\n')

print(' STATS FOR DIABETESPEDIGREEFUNCTION')
print(' DiabetesPedigreeFunction mean : ' + str(mean_DPF))
print(' DiabetesPedigreeFunction median : ' + str(median_DPF))
print(' DiabetesPedigreeFunction Standard Deviation : ' + str(std_DPF))
print(' DiabetesPedigreeFunction Range : ' + str(range_DPF))
print(' DiabetesPedigreeFunction 25th percentiles : ' + str(TwentyFivePer_DPF))
print(' DiabetesPedigreeFunction 50th percentiles : ' + str(FiftyPer_DPF))
print(' DiabetesPedigreeFunction 75th percentiles : ' + str(SeventyFivePer_DPF))
print('\n')

print(' STATS FOR AGE')
print(' Age mean : ' + str(mean_Age))
print(' Age median : ' + str(median_Age))
print(' Age Standard Deviation : ' + str(std_Age))
print(' Age Range : ' + str(range_Age))
print(' Age 25th percentiles : ' + str(TwentyFivePer_Age))
print(' Age 50th percentiles : ' + str(FiftyPer_Age))
print(' Age 75th percentiles : ' + str(SeventyFivePer_Age))
print('\n')

print(' STATS FOR CLASS')
print(' Class mean : ' + str(mean_Class))
print(' Class median : ' + str(median_Class))
print(' Class Standard Deviation : ' + str(std_Class))
print(' Class Range : ' + str(range_Class))
print(' Class 25th percentiles : ' + str(TwentyFivePer_Class))
print(' Class 50th percentiles : ' + str(FiftyPer_Class))
print(' Class 75th percentiles : ' + str(SeventyFivePer_Class))
print('\n')



#Part F
#Create histogram plot using 10 bins for the two features BloodPressure and DiabetesPedigreeFunction, respectively

# Histogram for BloodPressure using 10 bins
data.iloc[:,1].hist(bins =10)
plt.show()

# Histogram for DiabetesPedigreeFunction using 10 bins
data.iloc[:,4].hist(bins =10)
plt.show()



#Part G

#Quantile-quantile plot for BloodPressure
BP = data.iloc[:,1].values
scipy.stats.probplot(BP, dist = 'norm', plot= pylab)
pylab.show()

#Quantile-quantile plot for DiabetesPedigreeFunction
DPF = data.iloc[:,4].values
scipy.stats.probplot(DPF, dist = 'norm', plot= pylab)
pylab.show()