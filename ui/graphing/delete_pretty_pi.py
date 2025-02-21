import matplotlib.pyplot as plt
import numpy as np

def create_smart_pie_chart(sizes, labels, title=""):
    """
    Create a pie chart with intelligently placed labels that don't overlap.
    
    Parameters:
    sizes (list): Values for each pie slice
    labels (list): Labels for each slice
    title (str): Title of the pie chart
    """
    # Calculate the total to get percentages
    total = sum(sizes)
    percentages = [size/total * 100 for size in sizes]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get the center and radius of the pie chart
    center = (0, 0)
    radius = 1
    
    # Calculate the angles for each slice
    angles = [sum(percentages[:i])/100 * 2 * np.pi for i in range(len(percentages) + 1)]
    
    # Create the pie chart
    wedges, texts, autotexts = ax.pie(sizes,
                                     labels=labels,
                                     autopct='%1.1f%%',
                                     pctdistance=0.85,
                                     labeldistance=1.2)
    
    # Adjust label positions to prevent overlap
    # Get the positions of all labels
    label_positions = []
    for i, p in enumerate(wedges):
        ang = (angles[i] + angles[i+1])/2
        
        # Calculate the position for the label
        x = np.cos(ang) * radius * 1.2
        y = np.sin(ang) * radius * 1.2
        
        # Adjust position based on quadrant
        if x < 0:
            x *= 1.2  # Move left labels further left
        if y < 0:
            y *= 1.2  # Move bottom labels lower
            
        # Store position
        label_positions.append((x, y))
        
        # Set the label position
        texts[i].set_position((x, y))
        
        # Adjust percentage text position
        x_pct = np.cos(ang) * radius * 0.85
        y_pct = np.sin(ang) * radius * 0.85
        autotexts[i].set_position((x_pct, y_pct))
    
    # Set title
    if title:
        plt.title(title, pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Sample data
    sizes = [15, 30, 45, 10, 1, 1, 1, 1]
    labels = ['Category A', 'Category B', 'Category C', 'Category D'] * 2
    
    fig, ax = create_smart_pie_chart(sizes, labels, "Sample Pie Chart")
    plt.show()