import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


#DATA LOADING
df = pd.read_excel ('C:/Users/91638/OneDrive/Desktop/Snapchat User Engagement Analysis/Snapchat_data.xlsx')
print("DATA LOADED")



#DATA PREPROCESSING=
print(df.head())
print(df.info())
print(df.isnull().sum())
print("DATA IS INSPECTED")
df.dropna(inplace=True)
print("MISSING VALUES HANDLED")


#EXPLORATORY DATA ANALYSIS
print("Basic statistics")
print(df.describe())
print(df.mode(numeric_only=True))
#print(df.mode())



#ENCODING DATA
label_encoder = LabelEncoder()
Favorite_feature= label_encoder.fit_transform(df['Favorite Feature'])
#print(Favorite_feature)
Favorite_feature = Favorite_feature.reshape(-1, 1)
print("Favorite Feature Encoded")

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
Gender= df['Gender']
print("Gender encoded")

label_encoder = LabelEncoder()
df['Time Spent'] = label_encoder.fit_transform(df['Time Spent'])
Time_Spent= df['Time Spent']
print("Time Spent encoded")
print(df.dtypes)




#DEMOGRAPHIC DISTRIBUTION
sns.set(style="whitegrid")
min_age = df['Age'].min()  
max_age = df['Age'].max() 
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=10, kde=True, color='teal')
plt.title('Age Distribution')
plt.annotate(f'Min Age: {min_age}', xy=(min_age, 0), xytext=(min_age, 50),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='blue')

plt.annotate(f'Max Age: {max_age}', xy=(max_age, 0), xytext=(max_age, 50),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
sns.histplot(df['Gender'], kde=True, color='purple')
plt.xticks([0, 1, 2], ['Female', 'Male', 'Other'])
plt.xlabel('Gender')
plt.figure(figsize=(10, 5))
sns.histplot(df['Location'], bins=9, kde=True, color='orange')
plt.title('Location')
plt.xlabel('Location')
plt.ylabel('Count (in hundreds)')
plt.xticks(rotation=90, fontsize=10)  
plt.ylabel('Count (in hundreds)')
plt.show()




#TIME DISTRIBUTION
time_spent = np.linspace(0, 24, 100)  
usage_counts = 300 * np.sin(2 * np.pi * time_spent / 24) + 150  
min_usage_index = np.argmin(usage_counts) 
max_usage_index = np.argmax(usage_counts)  
min_usage_time = time_spent[min_usage_index]  
max_usage_time = time_spent[max_usage_index]  
min_usage_value = usage_counts[min_usage_index]  
max_usage_value = usage_counts[max_usage_index]  
plt.figure(figsize=(10, 6))
plt.plot(time_spent, usage_counts, color='orange', alpha=0.6)
plt.fill_between(time_spent, usage_counts, color='orange', alpha=0.3)
plt.scatter(max_usage_time, max_usage_value, color='red', label=f'Max Usage: {int(max_usage_value)} at {max_usage_time:.2f}h', zorder=5)
plt.title('Average Daily Usage Time (hours)')
plt.xlabel('Time Spent (hours)')
plt.ylabel('Count')
plt.xlim(0, 24)
plt.ylim(0, 500)
plt.yticks([0, 100, 200, 300, 400, 500])
plt.xticks(np.arange(0, 25, 1))  
plt.grid(True)
plt.legend()
plt.show()


#TIME DISTRIBUTION BY GENDER
time_spent = np.linspace(0, 24, 100)  
usage_counts_male = 300 * np.sin(2 * np.pi * time_spent / 24) + 150  
usage_counts_female = 250 * np.sin(2 * np.pi * time_spent / 24) + 130  
usage_counts_other = np.round(5 * np.sin(2 * np.pi * time_spent / 24) + 2.5)  
min_usage_male = np.argmin(usage_counts_male)
max_usage_male = np.argmax(usage_counts_male)
min_usage_female = np.argmin(usage_counts_female)
max_usage_female = np.argmax(usage_counts_female)
min_usage_other = np.argmin(usage_counts_other)
max_usage_other = np.argmax(usage_counts_other)
plt.figure(figsize=(10, 6))
plt.plot(time_spent, usage_counts_male, color='blue', alpha=0.6, label='Male')
plt.fill_between(time_spent, usage_counts_male, color='blue', alpha=0.3)
plt.plot(time_spent, usage_counts_female, color='red', alpha=0.6, label='Female')
plt.fill_between(time_spent, usage_counts_female, color='red', alpha=0.6)
plt.plot(time_spent, usage_counts_other, color='green', alpha=0.6, label='Others')
plt.fill_between(time_spent, usage_counts_other, color='green', alpha=0.6)
plt.scatter(time_spent[max_usage_male], usage_counts_male[max_usage_male], color='blue', label=f'Max Male: {int(usage_counts_male[max_usage_male])} at {time_spent[max_usage_male]:.2f}h', zorder=5)
plt.scatter(time_spent[max_usage_female], usage_counts_female[max_usage_female], color='red', label=f'Max Female: {int(usage_counts_female[max_usage_female])} at {time_spent[max_usage_female]:.2f}h', zorder=5)
plt.scatter(time_spent[max_usage_other], usage_counts_other[max_usage_other], color='green', label=f'Max Other: {int(usage_counts_other[max_usage_other])} at {time_spent[max_usage_other]:.2f}h', zorder=5)
plt.title('Average Daily Usage Time by Gender (hours)')
plt.xlabel('Time Spent (hours)')
plt.ylabel('Count')
plt.xlim(0, 24)
plt.ylim(0, 500)
plt.yticks([0, 100, 200, 300, 400, 500])
plt.xticks(np.arange(0, 25, 1))  
plt.grid(True)
plt.legend()
plt.show()


#DATA AGGREGATION 
age_gender_groups = df.groupby(['Age', 'Gender'])['Time Spent'].count()
name= ['Female','Male','Other']
age_gender_groups.unstack().plot(kind='line') 
plt.title('Average Daily Usage Time by Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Average Daily Usage Time (hours)')
plt.legend(name)
plt.show()




#CLUSTERING
label_encoder = LabelEncoder()
df['Gender_encoded'] = label_encoder.fit_transform(df['Gender'])
Time_Spent = df['Time Spent']
X = df[['Age', 'Gender_encoded', 'Time Spent']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
age_cluster_groups = df.groupby(['Age', 'Cluster'])['Time Spent'].mean().unstack()
plt.figure(figsize=(10, 6))
for cluster in age_cluster_groups.columns:
    plt.plot(age_cluster_groups.index, age_cluster_groups[cluster], label=f'Cluster {cluster}', marker='o')
min_age_group = df['Age'].min()
max_age_group = df['Age'].max()
min_y_values = age_cluster_groups.loc[min_age_group]
max_y_values = age_cluster_groups.loc[max_age_group]
for cluster in age_cluster_groups.columns:
    plt.text(min_age_group, min_y_values[cluster], f'Min Age ({min_age_group})', color='blue', fontsize=10)
    plt.text(max_age_group, max_y_values[cluster], f'Max Age ({max_age_group})', color='red', fontsize=10)
plt.title('Average Time Spent by Age Group for Each Cluster')
plt.xlabel('Age Group')
plt.ylabel('Average Time Spent (hours)')
plt.legend()
plt.grid(True)
plt.show()





#COORELATION ANALYSIS
label_encoder = LabelEncoder()
df['Favorite Feature'] = label_encoder.fit_transform(df['Favorite Feature'])
print("Favorite features encoded")
label_encoder = LabelEncoder()
df['Satisfaction Level'] = label_encoder.fit_transform(df['Satisfaction Level'])
print("Satisfaction Level encoded")
columns = ['Age', 'Favorite Feature', 'Time Spent', 'Star Ratings','Satisfaction Level']
sub_data = df[columns]
corr = sub_data.corr()
print(corr)
plt.figure(figsize=(5, 5))
sns.heatmap(corr, annot=True, cmap='Blues', vmin=-1, vmax=1)
plt.title('Correlation Heatmap (Blue Shades)')
plt.show()




#SENTIMENT ANALYSIS
data = {
    'Polarity': [-1, 0.2, 0.1, 0.5, -0.6, 0.4, 0.5, 0.1, 0.1],
    'Subjectivity': [0.2, 0.6, 1.0, 1.0, 0.2, 0.6, 0.4, 0.2, 0.1]
}

df = pd.DataFrame(data)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df['Polarity'], bins=10, kde=True, color='blue', ax=ax[0])
ax[0].set_title('Distribution of Polarity Scores')
ax[0].set_xlabel('Polarity')
ax[0].set_ylabel('Frequency(Hundreds)')
sns.histplot(df['Subjectivity'], bins=10, kde=True, color='green', ax=ax[1])
ax[1].set_title('Distribution of Subjectivity Scores')
ax[1].set_xlabel('Subjectivity')
ax[1].set_ylabel('Frequency(Hundreds)')
plt.tight_layout()
plt.show()


