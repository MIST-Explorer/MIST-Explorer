import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def BoxPlot():
    # Load your dataset from the Excel file
    file_path = r"C:\Users\meaha\Documents\Grouped Cells Biopsy Data.xlsx"
    data = pd.read_excel(file_path)

    # Define the region of interest
    x_min, y_min = 0, 0  # Adjust these coordinates as needed
    x_max, y_max = 12000, 12000

    # Filter the dataset to include only cells within the specified region
    filtered_data = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                        (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]

    # Extract the list of all proteins dynamically based on the column range
    protein_columns = filtered_data.columns[13:15]  # Assuming proteins start at the 4th column

    # Filter data to include only the protein expression columns
    protein_data = filtered_data[protein_columns]

    # Reshape the data for seaborn boxplot (melt to "long" format)
    protein_data_melted = protein_data.melt(var_name='Protein', value_name='Expression')

    # Plot the boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Protein', y='Expression', data=protein_data_melted, palette='Set2', showfliers=False)

    # Customize the plot
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Protein', fontsize=14)
    plt.ylabel('Expression Level', fontsize=14)
    plt.title(f'Protein Expression Box Plot for Region ({x_min}, {y_min}) to ({x_max}, {y_max})', fontsize=16)
    plt.subplots_adjust(bottom=0.25)
    # Display the plot
    plt.show()
