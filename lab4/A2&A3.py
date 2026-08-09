import pandas as pd
import time
import tracemalloc
df=pd.read_excel("ML-lab2 dataset.xlsx",sheet_name="marketing_campaign")
# logic computed by chatgpt
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]) 
df1=df.copy()
#date is not categorical but it is stored as a string
#from A1 after identifying that date is stored as string we change it to this
def labelEncoding(df1, column):
    data = df1.copy()

    values = data[column].dropna().unique()

    labels = {}

    # Assign numbers starting from 0
    for i, value in enumerate(values):
        labels[value] = i

    data[column] = data[column].map(labels)

    return data, labels

def oneHot(df1, column):
    data = df1.copy()

    # Get unique values
    values = data[column].dropna().unique()

    # Create a new column for each unique value
    for value in values:
        data[str(value)] = (data[column] == value).astype(int)

    # Remove original column
    data.drop(column, axis=1, inplace=True)

    return data

def performance_test(func, *args):

    # Start measuring memory
    tracemalloc.start()

    # Start timer
    start = time.perf_counter()

    # Execute the function
    result = func(*args)

    # Stop timer
    end = time.perf_counter()

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()

    # Stop memory tracking
    tracemalloc.stop()

    print("Execution Time:", end - start, "seconds")
    print("Peak Memory:", peak / 1024, "KB")

    return result

####################### A2 ###############################
# --------------------------
# Label Encoding Example
# --------------------------

df1, labels = labelEncoding(df1, "Education")

print("Encoded Labels:")
print(labels)

label_df=df1.copy()
# --------------------------
# One Hot Encoding Example
# --------------------------

df1 = oneHot(df1, "Marital_Status")


########################### A3 #######################
print("Original Dataset Shape:", df.shape)
print("Encoded Dataset Shape:", df1.shape)
print()
print()# to separate from unit test cases

#       UNIT TEST CASE SECTION
# changing test case to a different example as chatgpt gave the same feature from excel
########################### A2 ###########################
print("UNIT TEST CASE SECTION")
print()
test_df1 = pd.DataFrame({
    "sleep_quality": ["poor", "fair", "good", "excellent", "fair", "excellent"]
})

# Run Function
encoded_df1, labels = labelEncoding(test_df1, "sleep_quality")

# Display Results
print("Original Data:")
print(test_df1)

print("\nEncoded Data:")
print(encoded_df1)

print("\nLabels Dictionary:")
print(labels)

test_df1 = pd.DataFrame({
    "Color": ["Red", "Blue", "Green", "Red", "Blue"]
})

# Run Function
encoded_df1 = oneHot(test_df1, "Color")

# Display Results
print("Original Data:")
print(test_df1)

print("\nOne-Hot Encoded Data:")
print(encoded_df1)

###################### A3 #########################################
# -------------------------------
# Unit Test: Label Encoding Shape
# -------------------------------

assert label_df.shape == df.shape

print("Label Encoding Dimension Test: Passed")

# --------------------------------
# Unit Test: One-Hot Encoding Shape
# --------------------------------

unique_values = df["Marital_Status"].nunique()

expected_columns = df.shape[1] - 1 + unique_values

assert df1.shape == (df.shape[0], expected_columns)

print("One-Hot Encoding Dimension Test: Passed")

print("performance for label encoding")
print(performance_test(labelEncoding,df1,"Education"))
print("performance for one-hot encoding")
print(performance_test(oneHot,label_df,"Marital_Status"))


